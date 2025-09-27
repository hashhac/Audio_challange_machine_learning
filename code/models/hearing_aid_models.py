# hearing_aid_models.py

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
import soundfile as sf

# --- 1. SETUP: IMPORTS AND ROBUST PATHING ---
def setup_paths():
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # Go up one more level to reach project root
    code_path = project_root / "code"
    
    # Add both code path and parent directory to Python path
    if str(code_path) not in sys.path:
        sys.path.append(str(code_path))
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    
    logging.info(f"Project root is: {project_root}")
    logging.info(f"Added to Python path: {code_path}")
    return project_root, code_path

project_root, code_path = setup_paths()

# Import using relative path from code directory
sys.path.insert(0, str(code_path))  # Ensure code directory is searched first
from data_loading import create_dataloaders
from clarity.enhancer.nalr import NALR
from clarity.enhancer.compressor import Compressor
from clarity.utils.audiogram import Audiogram

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# The audiogram was not provided in the metadata so where using these values for a test run we do not have the actual audiogram values
DEFAULT_AUDIOGRAMS = {
    "No Loss": [ -5., -5., -5., -5., -5., -5.],
    "Mild": [ 20., 25., 30., 40., 50., 55.],
    "Moderate": [ 45., 50., 60., 65., 70., 75.],
    "Moderately Severe": [ 65., 70., 75., 80., 85., 90.],
}
# --- 2. THE REFINED MODEL CLASS (NOW NAMED NLR1 AS YOU SUGGESTED) ---
class NLR1_Model:
    """
    Our first baseline enhancement model (NLR1) using PyClarity's NAL-R and Compressor.
    """
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.nalr = NALR(nfir=220, sample_rate=self.sample_rate)
        # Compressor expects 'fs' argument name; pass it explicitly
        self.compressor = Compressor(
            fs=self.sample_rate, attack=5.0, release=50.0,
            threshold=0.0, attenuation=0.0001, makeup_gain=1.0
        )

    def process_audio(self, signal_tensor: torch.Tensor, audiogram_levels: list) -> torch.Tensor:
        signal_np = signal_tensor.cpu().numpy()
        audiogram = Audiogram(levels=audiogram_levels, frequencies=[250, 500, 1000, 2000, 4000, 6000])
        if signal_np.ndim == 1:
            signal_np = np.stack([signal_np, signal_np])
        processed_channels = np.zeros_like(signal_np)
        # Build NAL-R filter for this audiogram
        nalr_filter, delay = self.nalr.build(audiogram)
        assert nalr_filter is not None, "NAL-R filter build failed"

        for i in range(signal_np.shape[0]):
            channel = signal_np[i, :]
            assert channel.ndim == 1, "Expected 1D channel signal"

            # Apply NAL-R filtering
            nalr_applied = self.nalr.apply(nalr_filter, channel)

            # Trim or pad to original length: convolution may change length
            orig_len = channel.shape[0]
            if nalr_applied.shape[0] >= orig_len:
                nalr_applied = nalr_applied[:orig_len]
            else:
                # pad with zeros
                padded = np.zeros(orig_len, dtype=nalr_applied.dtype)
                padded[: nalr_applied.shape[0]] = nalr_applied
                nalr_applied = padded

            # Compression
            compressed_signal, _, _ = self.compressor.process(nalr_applied)
            processed_channels[i, :] = compressed_signal
        return torch.from_numpy(processed_channels).float()


# --- 3. THE MAIN "WORKER" FUNCTION ---
def enhance_and_save_dataset(dataloader: DataLoader, output_dir: Path):
    model = NLR1_Model()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Starting enhancement process. Output will be saved to: {output_dir}")

    for batch in tqdm(dataloader, desc="Enhancing Audio Files"):
        signals = batch['signal']
        # This is the batched list of metadata dictionaries
        metadata_list = batch['metadata'] 
        
        for i in range(len(signals)):
            # --- THE KEYERROR FIX IS HERE ---
            current_metadata = metadata_list[i] # First, get the dictionary for this item

            # print(f"DEBUG: Metadata keys are: {current_metadata.keys()}")
            hearing_loss_label = current_metadata.get('hearing_loss', "No Loss")
                
            # 2. Look up the correct audiogram from our local dictionary
            audiogram = DEFAULT_AUDIOGRAMS.get(hearing_loss_label, DEFAULT_AUDIOGRAMS["No Loss"])
            # -------------------------------------------------
            
            signal_id = current_metadata['signal']
            # --------------------------------

            processed_tensor = model.process_audio(signals[i], audiogram)
            output_filename = output_dir / f"{signal_id}.wav"
            processed_numpy = processed_tensor.cpu().numpy().T
            sf.write(output_filename, processed_numpy, model.sample_rate)


# --- 4. A RUNNABLE EXAMPLE FOR TESTING ---
if __name__ == '__main__':
    logging.info("--- Running base_line.py as a standalone script for testing ---")
    
    # Your idea: Define a specific output directory for this model
    # This keeps results organized and separate for each model version.
    model_name = "NLR1"
    output_directory = project_root / "results" / model_name
    
    cadenza_data_root = project_root / "code" / "audio_files" / "16981327" / "cadenza_data"
    _, valid_loader = create_dataloaders(cadenza_data_root.parent, batch_size=4)
    
    enhance_and_save_dataset(valid_loader, output_directory)
    
    logging.info(f"--- Standalone test complete. Check the output folder: {output_directory} ---")