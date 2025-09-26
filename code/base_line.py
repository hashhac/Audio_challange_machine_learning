# base_line.py

import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import logging
import soundfile as sf

# --- 1. SETUP: IMPORTS AND ROBUST PATHING ---

# Add project root to path to allow importing from other folders
def setup_paths():
    """Setup all necessary paths to find our dataloader."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent # Assumes this file is in .../code/
    
    # Add the 'code' directory to the path to find data_loading.py
    code_path = project_root / "code"
    if str(code_path) not in sys.path:
        sys.path.append(str(code_path))
        
    logging.info(f"Project root is: {project_root}")
    logging.info(f"Added to Python path: {code_path}")
    
    return project_root

project_root = setup_paths()

# Now we can safely import our custom modules
from data_loading import create_dataloaders
from clarity.enhancer.nalr import NALR
from clarity.enhancer.compressor import Compressor
from clarity.utils.audiogram import Audiogram

# Configure logging for clean output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 2. THE REFINED MODEL CLASS ---

class BaselineModel:
    """
    Baseline enhancement model using PyClarity's NALR and Compressor.
    Refactored for clarity and robustness.
    """
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.nalr = NALR(nfir=220, sample_rate=self.sample_rate)
        self.compressor = Compressor(
            sample_rate=self.sample_rate, attack=5e-3, release=50e-3,
            threshold=0.0, ratio=1.0
        )

    def process_audio(self, signal_tensor: torch.Tensor, audiogram_levels: list) -> torch.Tensor:
        """
        Processes a stereo audio signal using the NAL-R prescription.
        Handles each channel independently.
        """
        # Convert torch tensor to numpy array for pyclarity processing
        signal_np = signal_tensor.cpu().numpy()
        
        # Create the audiogram object needed by NAL-R
        audiogram = Audiogram(levels=audiogram_levels, frequencies=[250, 500, 1000, 2000, 4000, 6000])
        
        # Ensure we have a 2D array for stereo processing
        if signal_np.ndim == 1:
            signal_np = np.stack([signal_np, signal_np]) # Convert mono to stereo

        # Create an empty array to store the processed channels
        processed_channels = np.zeros_like(signal_np)

        # Process each channel of the stereo signal independently
        for i in range(signal_np.shape[0]):
            nalr_processed = self.nalr.process(signal=signal_np[i, :], audiogram=audiogram)
            compressed_signal, _, _ = self.compressor.process(nalr_processed)
            processed_channels[i, :] = compressed_signal
            
        return torch.from_numpy(processed_channels).float()


# --- 3. THE MAIN "WORKER" FUNCTION ---

def enhance_and_save_dataset(dataloader: DataLoader, output_dir: Path):
    """
    Loops through a dataloader, enhances each audio file using the
    BaselineModel, and saves the output to a specified directory.
    """
    model = BaselineModel()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting enhancement process. Output will be saved to: {output_dir}")

    for batch in tqdm(dataloader, desc="Enhancing Audio Files"):
        signals = batch['signal']
        # The dataloader batches metadata into a dictionary of lists.
        # We access this dictionary directly.
        metadata_batch = batch['metadata']
        
        for i in range(len(signals)):
            # --- THE KEYERROR FIX IS HERE ---
            # Instead of trying to rebuild a 'metadata' dict for each item,
            # we directly access the lists within the batched metadata.
            current_signal_tensor = signals[i]
            audiogram = metadata_batch['audiogram_levels_l'][i]
            signal_id = metadata_batch['signal'][i]
            # --------------------------------

            # Use the model to process the audio
            processed_tensor = model.process_audio(current_signal_tensor, audiogram)
            
            # Define the output filename
            output_filename = output_dir / f"{signal_id}.wav"
            
            # Convert tensor to numpy and save the .wav file
            processed_numpy = processed_tensor.cpu().numpy().T
            sf.write(output_filename, processed_numpy, model.sample_rate)


# --- 4. A RUNNABLE EXAMPLE FOR TESTING ---
# This block allows us to run this script directly to test the enhancement
# process without needing the full run_experiment.py script.

if __name__ == '__main__':
    logging.info("--- Running baseline.py as a standalone script for testing ---")
    
    # Define the necessary paths starting from our robust project_root
    cadenza_data_root = project_root / "code" / "audio_files" / "16981327" / "cadenza_data"
    output_directory = project_root / "results" / "my_enhanced_audio_test"
    
    # Create a dataloader for the validation set
    # We pass the PARENT of 'cadenza_data' to the function
    _, valid_loader = create_dataloaders(cadenza_data_root.parent, batch_size=4)
    
    # Run the main enhancement function
    enhance_and_save_dataset(valid_loader, output_directory)
    
    logging.info("--- Standalone test complete. Check the output folder. ---")