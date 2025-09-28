# hearing_aid_models.py

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import logging
import soundfile as sf

# --- 1. SETUP: IMPORTS AND ROBUST PATHING ---
# (This section is perfect and remains unchanged)
def setup_paths():
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    code_path = project_root / "code"
    if str(code_path) not in sys.path:
        sys.path.append(str(code_path))
    logging.info(f"Project root is: {project_root}")
    logging.info(f"Added to Python path: {code_path}")
    return project_root
project_root = setup_paths()

from data_loading import create_dataloaders
from clarity.enhancer.nalr import NALR
from clarity.enhancer.compressor import Compressor
from clarity.utils.audiogram import Audiogram

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_AUDIOGRAMS = {
    "No Loss": [ -5., -5., -5., -5., -5., -5.],
    "Mild": [ 20., 25., 30., 40., 50., 55.],
    "Moderate": [ 45., 50., 60., 65., 70., 75.],
    "Moderately Severe": [ 65., 70., 75., 80., 85., 90.],
}

# --- 2. THE CORRECTED AND SIMPLIFIED MODEL CLASS ---
class NLR1_Model:
    """
    Our first baseline enhancement model (NLR1), corrected to produce non-silent audio.
    """
    def __init__(self, sample_rate=44100, audiogram_levels: list = None):
        """
        sample_rate: sample rate used by NALR/Compressor
        audiogram_levels: optional default audiogram levels used when per-file metadata lacks audiogram info
        """
        self.sample_rate = sample_rate
        # NALR expects (nfir, sample_rate)
        self.nalr = NALR(nfir=220, sample_rate=self.sample_rate)
        # Compressor expects fs (Hz) and attack/release in milliseconds.
        # Set attenuation to a reasonable value (0.5) instead of tiny 0.0001 to avoid muting.
        self.compressor = Compressor(
            fs=self.sample_rate,
            attack=5.0,      # ms
            release=50.0,    # ms
            threshold=0.0,
            attenuation=0.5,
            makeup_gain=1.0,
        )

        # Use provided audiogram or the default "No Loss"
        if audiogram_levels is None:
            self.default_audiogram = DEFAULT_AUDIOGRAMS["No Loss"]
        else:
            self.default_audiogram = audiogram_levels

    def _extract_audiogram_from_metadata(self, metadata: dict):
        """Return audiogram levels list from metadata with sensible fallbacks."""
        if metadata is None:
            return self.default_audiogram
        # Common variants we may encounter
        for key in ("audiogram_levels_l", "audiogram_l", "audiogram", "hearing_loss"):
            if key in metadata:
                val = metadata[key]
                # If hearing_loss label like 'Mild' map to default table
                if key == "hearing_loss":
                    return DEFAULT_AUDIOGRAMS.get(val, self.default_audiogram)
                # If it's already a list or tuple of numbers
                if isinstance(val, (list, tuple)):
                    return list(val)
                # If it's a string representation try to parse numbers
                if isinstance(val, str):
                    try:
                        parts = [float(x) for x in val.replace(',', ' ').split()]
                        if parts:
                            return parts
                    except Exception:
                        pass
        return self.default_audiogram

    def process_audio(self, signal_tensor: torch.Tensor, audiogram_levels: list = None, metadata: dict = None) -> torch.Tensor:
        signal_np = signal_tensor.cpu().numpy()
        # Ensure shape is (channels, samples)
        if signal_np.ndim == 1:
            signal_np = np.stack([signal_np, signal_np])
        assert signal_np.ndim == 2, "signal must be (channels, samples)"

        # Determine audiogram: parameter takes precedence, then metadata, then instance default
        if audiogram_levels is None:
            audiogram_levels = self._extract_audiogram_from_metadata(metadata)

        audiogram = Audiogram(levels=audiogram_levels, frequencies=[250, 500, 1000, 2000, 4000, 6000])

        # Build NAL-R filter once per audiogram
        nalr_filter, nalr_delay = self.nalr.build(audiogram)
        processed_channels = np.zeros_like(signal_np)

        for ch in range(signal_np.shape[0]):
            # apply NAL-R filter to the channel
            nalr_applied = self.nalr.apply(nalr_filter, signal_np[ch, :])

            # nalr.apply may change length (convolution). Trim or pad to original length.
            orig_len = signal_np.shape[1]
            out_len = nalr_applied.shape[0]
            if out_len > orig_len:
                start = (out_len - orig_len) // 2
                nalr_applied = nalr_applied[start:start + orig_len]
            elif out_len < orig_len:
                nalr_applied = np.pad(nalr_applied, (0, orig_len - out_len))

            assert nalr_applied.shape[0] == orig_len, "NALR output length mismatch"

            # Compress
            compressed_signal, _, _ = self.compressor.process(nalr_applied.astype(np.float32))

            # Ensure shape matches and store
            assert compressed_signal.shape[0] == orig_len, "Compressor output length mismatch"
            processed_channels[ch, :] = compressed_signal

        return torch.from_numpy(processed_channels).float()


# --- 3. THE MAIN "WORKER" FUNCTION ---
def enhance_and_save_dataset(dataloader: DataLoader, output_dir: Path, debug_pairs: int = 0):
    model = NLR1_Model()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Starting enhancement process. Output will be saved to: {output_dir}")

    # Re-introducing the robust normalize and clip function as a safety measure
    def normalize_and_clip(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
        x = x.astype('float32')
        max_abs = float(np.max(np.abs(x)) + 1e-12)
        if max_abs > 0:
            x = x / max_abs * peak
        np.clip(x, -1.0, 1.0, out=x)
        return x

    debug_count = 0

    for batch in tqdm(dataloader, desc="Enhancing Audio Files"):
        signals = batch['signal']
        metadata_list = batch['metadata']

        for i in range(len(signals)):
            # Prepare signal id for logging even if an exception occurs
            signal_id = None
            try:
                current_metadata = metadata_list[i]
                signal_id = current_metadata['signal']
                hearing_loss_label = current_metadata.get('hearing_loss', "No Loss")
                audiogram = DEFAULT_AUDIOGRAMS.get(hearing_loss_label, DEFAULT_AUDIOGRAMS["No Loss"])

                processed_tensor = model.process_audio(signals[i], audiogram)

                # Original audio (prepare for saving): ensure (samples, channels)
                orig_numpy = signals[i].cpu().numpy()
                if orig_numpy.ndim == 1:
                    orig_numpy = orig_numpy.reshape(1, -1)
                orig_to_write = orig_numpy.T.astype('float32')

                # Processed audio
                processed_numpy = processed_tensor.cpu().numpy()
                if processed_numpy.ndim == 1:
                    processed_numpy = processed_numpy.reshape(1, -1)
                proc_to_write = processed_numpy.T.astype('float32')

                # Apply safety normalization to processed output
                proc_to_write = normalize_and_clip(proc_to_write)

                if debug_count < debug_pairs:
                    idx = debug_count + 1
                    before_path = output_dir / f"{idx:02d}_before_{signal_id}.wav"
                    after_path = output_dir / f"{idx:02d}_after_{signal_id}.wav"
                    logging.info(f"Saving debug pair {idx}: {before_path.name}, {after_path.name}")
                    sf.write(before_path, orig_to_write, model.sample_rate)
                    sf.write(after_path, proc_to_write, model.sample_rate)
                    debug_count += 1
                else:
                    output_filename = output_dir / f"{signal_id}.wav"
                    sf.write(output_filename, proc_to_write, model.sample_rate)

            except Exception as e:
                logging.error(f"An unexpected error occurred for signal {signal_id or 'unknown'}: {e}. Skipping.")
                raise

# --- 4. A FAST, 10-FILE TEST RUN ---
if __name__ == '__main__':
    logging.info("--- Running base_line.py in FAST TEST mode (10 files) ---")

    # Setup paths at runtime (avoid doing this at import time)
    project_root = setup_paths()

    model_name = "NLR1_test_run"
    output_directory = project_root / "results" / model_name

    cadenza_data_root = project_root / "code" / "audio_files" / "16981327" / "cadenza_data"

    # We create the full dataloader first
    _, full_valid_loader = create_dataloaders(cadenza_data_root.parent, batch_size=4)

    # --- THIS IS THE TRICK FOR A SMALL TEST ---
    # We use PyTorch's Subset to create a mini-dataset with only the first 10 samples
    num_test_samples = 10
    subset_indices = list(range(num_test_samples))
    test_dataset = Subset(full_valid_loader.dataset, subset_indices)

    # Now create a new dataloader that only uses our mini-dataset
    test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=full_valid_loader.collate_fn)
    # -------------------------------------------

    enhance_and_save_dataset(test_loader, output_directory, debug_pairs=10)

    logging.info(f"--- Fast test complete. Check the 10 audio files in: {output_directory} ---")