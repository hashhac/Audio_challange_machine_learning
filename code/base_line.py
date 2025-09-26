import torch
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging
import soundfile as sf
path_to_baseline_code = Path(r"Audio_challange_machine_learning\code\audio_files\16981327\clarity\recipes\cad_icassp_2026\baseline")
# Corrected imports from pyclarity
from clarity.enhancer.nalr import NALR
from clarity.enhancer.compressor import Compressor
from clarity.utils.audiogram import Audiogram
# Make sure your dataloading script is in the same directory or accessible
from data_loading import create_dataloaders 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaselineModel:
    """Baseline enhancement model using PyClarity's NALR and Compressor."""
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        # FIX: Provide the required 'nfir' and 'sample_rate' arguments
        self.nalr = NALR(nfir=220, sample_rate=self.sample_rate)
        # The compressor is also a key part of a hearing aid model
        self.compressor = Compressor(
            sample_rate=self.sample_rate,
            # These are example parameters, you can tune them
            attack=5e-3,
            release=50e-3,
            threshold=0.0,
            ratio=1.0
        )

    def process_audio(self, signal, audiogram_levels):
        """Process an audio signal using NAL-R for a given audiogram."""
        # Ensure audio is a NumPy array for pyclarity processing
        if isinstance(signal, torch.Tensor):
            # Squeeze to remove channel dimension if it's mono [1, N] -> [N]
            signal = signal.squeeze().numpy()
        
        # The NALR enhancer expects the audiogram levels as input
        # We need to create the 'audiogram' object pyclarity expects
        
        audiogram = Audiogram(levels=audiogram_levels, frequencies=[250, 500, 1000, 2000, 4000, 6000])

        # Apply NAL-R enhancement based on the listener's specific hearing loss
        nalr_processed = self.nalr.process(signal=signal, audiogram=audiogram)
        
        # Apply compression
        compressed_signal, _, _ = self.compressor.process(nalr_processed)
        
        return torch.from_numpy(compressed_signal).float()

    def calculate_metrics(self, processed, target):
        """Calculate a simple performance metric (e.g., correlation)."""
        # Ensure tensors are flat for correlation
        processed_flat = processed.flatten()
        target_flat = target.flatten()
        
        # To avoid errors with silent signals, check the standard deviation
        if torch.std(processed_flat) < 1e-6 or torch.std(target_flat) < 1e-6:
            return {'correlation': 0.0}

        correlation = torch.corrcoef(
            torch.stack([processed_flat, target_flat])
        )[0, 1]
        
        return {'correlation': correlation.item()}

def evaluate_baseline(data_dir: Path, output_dir: Path, batch_size: int = 4):
    """Evaluate the baseline enhancement model AND SAVE THE OUTPUT."""
    model = BaselineModel()
    _, valid_loader = create_dataloaders(data_dir, batch_size=batch_size) # Ensure you have a collate_fn
    
    # --- THIS IS THE FOLDER WE NEED TO CREATE ---
    enhanced_audio_output_dir = output_dir / "my_enhanced_audio"
    enhanced_audio_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Enhanced audio will be saved to: {enhanced_audio_output_dir}")
    
    results = []
    
    logging.info("Evaluating baseline model and SAVING enhanced audio...")
    for batch in tqdm(valid_loader):
        signals = batch['signal']
        unprocessed = batch['unprocessed']
        metadata_list = batch['metadata']
        
        for i in range(len(signals)):
            metadata = metadata_list[i]
            audiogram_levels = metadata['audiogram_levels_l']
            
            # This is your enhancement step
            processed_signal_tensor = model.process_audio(signals[i], audiogram_levels)
            
            # --- ADD THIS CODE TO SAVE THE FILE ---
            signal_id = metadata['signal']
            output_filename = enhanced_audio_output_dir / f"{signal_id}.wav"
            
            # Convert tensor to numpy array for saving
            # The .T transposes the data to the shape soundfile expects (samples, channels)
            processed_signal_numpy = processed_signal_tensor.cpu().numpy().T
            
            # Save the processed audio as a WAV file
            sf.write(output_filename, processed_signal_numpy, model.sample_rate)
            # ----------------------------------------

            # The rest of your script can continue as before
            metrics = model.calculate_metrics(processed_signal_tensor, unprocessed[i])
            results.append({
                'signal_id': signal_id,
                **metrics
            })
    
    # Create results DataFrame and save
    results_df = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'enhancement_baseline_results.csv', index=False)
    
    logging.info("\nValidation Set Results:")
    logging.info(f"Average correlation: {results_df['correlation'].mean():.3f}")
    
    return results_df

if __name__ == "__main__":
    # Setup paths
    current_dir = Path(__file__).parent
    # The data_dir should point to the directory containing 'cadenza_data'
    data_dir = current_dir / "audio_files" / "16981327"
    output_dir = current_dir / "results"
    
    logging.info("Starting baseline evaluation...")
    results = evaluate_baseline(data_dir, output_dir)
    logging.info("Evaluation complete. Results saved to enhancement_baseline_results.csv")