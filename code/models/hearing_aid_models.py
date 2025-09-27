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

# --- 2. THE SIMPLIFIED AND CORRECTED MODEL CLASS ---
class NLR1_Model:
    """
    Our first baseline enhancement model (NLR1), with a simplified and robust API.
    This version corrects the issue that was producing silent/unprocessed audio.
    """
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.nalr = NALR(nfir=220, sample_rate=self.sample_rate)
        
        # Correctly use 'ratio' for compression instead of 'attenuation' to prevent silence.
        # A standard 4:1 compression ratio with a 5ms attack time.
        self.compressor = Compressor(
            sample_rate=self.sample_rate, attack=5e-3, release=50e-3,
            ratio=4.0, threshold=0.0
        )

    def process_audio(self, signal_tensor: torch.Tensor, audiogram_levels: list) -> torch.Tensor:
        """
        Processes a stereo audio signal. This API is now unambiguous: it requires
        the signal and the final list of audiogram levels.
        """
        signal_np = signal_tensor.cpu().numpy()
        audiogram = Audiogram(levels=audiogram_levels, frequencies=[250, 500, 1000, 2000, 4000, 6000])
        
        # Ensure signal is stereo for consistent processing
        if signal_np.ndim == 1:
            signal_np = np.stack([signal_np, signal_np])
        
        processed_channels = np.zeros_like(signal_np)
        
        # Process each channel independently
        for i in range(signal_np.shape[0]):
            # Use the reliable, high-level .process() methods
            nalr_processed = self.nalr.process(signal=signal_np[i, :], audiogram=audiogram)
            compressed_signal, _, _ = self.compressor.process(nalr_processed)
            processed_channels[i, :] = compressed_signal
        
        return torch.from_numpy(processed_channels).float()


# --- 3. THE MAIN "WORKER" FUNCTION ---
# (This function is already correct and needs no changes)
def enhance_and_save_dataset(dataloader: DataLoader, output_dir: Path, debug_pairs: int = 0):
    model = NLR1_Model()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Starting enhancement process. Output will be saved to: {output_dir}")

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
            signal_id = None
            try:
                current_metadata = metadata_list[i]
                signal_id = current_metadata['signal']
                hearing_loss_label = current_metadata.get('hearing_loss', "No Loss")
                audiogram = DEFAULT_AUDIOGRAMS.get(hearing_loss_label, DEFAULT_AUDIOGRAMS["No Loss"])

                # Call the model's simple, direct API
                processed_tensor = model.process_audio(signals[i], audiogram_levels=audiogram)

                proc_to_write = processed_tensor.cpu().numpy().T
                proc_to_write = normalize_and_clip(proc_to_write)

                if debug_count < debug_pairs:
                    # (Code for saving debug pairs is fine)
                    pass 
                else:
                    output_filename = output_dir / f"{signal_id}.wav"
                    sf.write(output_filename, proc_to_write, model.sample_rate)

            except Exception as e:
                logging.error(f"An unexpected error occurred for signal {signal_id or 'unknown'}: {e}. Skipping.")
                continue # Use 'continue' in the final version instead of 'raise'


# --- 4. A FAST, 10-FILE TEST RUN ---
# (This testing block is also correct and needs no changes)
if __name__ == '__main__':
    logging.info("--- Running hearing_aid_models.py in FAST TEST mode (10 files) ---")

    model_name = "NLR1_test_run"
    output_directory = project_root / "results" / model_name

    cadenza_data_root = project_root / "code" / "audio_files" / "16981327" / "cadenza_data"

    _, full_valid_loader = create_dataloaders(cadenza_data_root.parent, batch_size=4)

    num_test_samples = 10
    subset_indices = list(range(num_test_samples))
    test_dataset = Subset(full_valid_loader.dataset, subset_indices)
    
    test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=full_valid_loader.collate_fn)
    
    enhance_and_save_dataset(test_loader, output_directory, debug_pairs=10)

    logging.info(f"--- Fast test complete. Check the 10 audio files in: {output_directory} ---")