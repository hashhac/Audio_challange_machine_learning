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
    """Evaluate the baseline enhancement model on the dataset."""
    # Note: Using a smaller batch_size can help with memory issues
    model = BaselineModel()
    # Your create_dataloaders will give you both train and validation loaders
    _, valid_loader = create_dataloaders(data_dir, batch_size=batch_size)
    
    results = []
    
    logging.info("Evaluating baseline model on validation set...")
    for batch in tqdm(valid_loader):
        signals = batch['signal'] # Now a padded tensor [batch, channels, length]
        unprocessed = batch['unprocessed'] # Also a padded tensor
        metadata_list = batch['metadata'] # This is a LIST of dictionaries
        
        for i in range(len(signals)):
            # Get the metadata for the i-th item in the batch
            metadata = metadata_list[i]
            
            # Important: Get the original, unpadded length if needed for processing
            # This is a good practice, though NALR might not need it.
            # original_length = metadata['original_length'] # You would add this in your dataset __getitem__

            audiogram_levels = metadata['audiogram_levels_l'] # Access the dict directly
            
            # Select the i-th signal from the batch
            current_signal = signals[i]
            
            # Process the audio
            processed_signal = model.process_audio(current_signal, audiogram_levels)
                
            # Calculate metrics comparing the enhanced signal to the original unprocessed one
            metrics = model.calculate_metrics(processed_signal, unprocessed[i])
            
            # Store results
            results.append({
                'signal_id': metadata['signal'][i],
                **metrics,
                'hearing_loss': metadata['hearing_loss'][i]
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