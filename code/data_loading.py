import os
import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import logging
from torch.nn.utils.rnn import pad_sequence


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load metadata from JSON file into a pandas DataFrame"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return pd.DataFrame(metadata)

class CadenzaDataset(Dataset):
    """Dataset for Cadenza audio data with metadata integration"""
    def __init__(self, root_dir: str, split: str = 'train'):
        """
        Initialize the dataset
        Args:
            root_dir: Path to the cadenza_data directory
            split: 'train' or 'valid'
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.signals_dir = self.root_dir / split / 'signals'
        self.unprocessed_dir = self.root_dir / split / 'unprocessed'
        
        # Load metadata
        metadata_path = self.root_dir / 'metadata' / f'{split}_metadata.json'
        self.metadata_df = load_metadata(metadata_path)
        
        # Get all audio files (supporting both .flac and .wav)
        self.signal_files = []
        self.unprocessed_files = []
        
        # Map metadata signal IDs to actual files
        for signal_id in self.metadata_df['signal']:
            signal_file = next(self.signals_dir.glob(f"{signal_id}.*"))
            unproc_file = next(self.unprocessed_dir.glob(f"{signal_id}_unproc.*"))
            self.signal_files.append(signal_file)
            self.unprocessed_files.append(unproc_file)

    def __len__(self) -> int:
        return len(self.signal_files)

    def __getitem__(self, idx: int) -> Dict:
        signal_path = str(self.signal_files[idx])
        unprocessed_path = str(self.unprocessed_files[idx])
        
        # Load audio files
        signal, sr = torchaudio.load(signal_path)
        unprocessed, _ = torchaudio.load(unprocessed_path)
        
        # Get metadata for this item
        metadata = self.metadata_df.iloc[idx].to_dict()
        
        return {
            'signal': signal,
            'unprocessed': unprocessed,
            'sample_rate': sr,
            'signal_path': signal_path,
            'unprocessed_path': unprocessed_path,
            'metadata': metadata
        }

def pad_collate(batch):
    """
    Pads audio signals in a batch to the same length.
    Assumes batch is a list of dictionaries.
    """
    # Separate the components of the batch
    signals = [item['signal'].T for item in batch]  # Transpose to (Length, Channels) for padding
    unprocessed = [item['unprocessed'].T for item in batch]
    
    # Use pad_sequence to pad the signals. batch_first=True makes the output (Batch, Length, Channels)
    signals_padded = pad_sequence(signals, batch_first=True, padding_value=0.0)
    unprocessed_padded = pad_sequence(unprocessed, batch_first=True, padding_value=0.0)

    # We need to transpose back to (Batch, Channels, Length) for PyTorch conventions
    signals_padded = signals_padded.permute(0, 2, 1)
    unprocessed_padded = unprocessed_padded.permute(0, 2, 1)

    # We also need to gather the other data. The default collate can handle most of it.
    # Let's manually collect metadata and paths for clarity.
    sample_rates = [item['sample_rate'] for item in batch]
    signal_paths = [item['signal_path'] for item in batch]
    unprocessed_paths = [item['unprocessed_path'] for item in batch]
    metadata = [item['metadata'] for item in batch]

    # Reconstruct the batch as a single dictionary
    return {
        'signal': signals_padded,
        'unprocessed': unprocessed_padded,
        'sample_rate': sample_rates,
        'signal_path': signal_paths,
        'unprocessed_path': unprocessed_paths,
        'metadata': metadata # This will now be a list of dicts
    }
def get_dataset_stats(dataset_dir: Path) -> Dict:
    """
    Get comprehensive statistics about the dataset including metadata analysis
    Returns a dictionary containing all statistics
    """
    stats = {}
    dataset_dir = Path(dataset_dir)
    
    for split in ['train', 'valid']:
        # Load metadata
        metadata_path = dataset_dir / 'metadata' / f'{split}_metadata.json'
        if metadata_path.exists():
            metadata_df = load_metadata(metadata_path)
            
            # Basic file counts
            signals_dir = dataset_dir / split / 'signals'
            unprocessed_dir = dataset_dir / split / 'unprocessed'
            n_signals = len(list(signals_dir.glob('*.flac')))
            n_unprocessed = len(list(unprocessed_dir.glob('*.flac')))
            
            print(f"\n{split.capitalize()} Dataset Statistics:")
            print(f"Number of signal files: {n_signals}")
            print(f"Number of unprocessed files: {n_unprocessed}")
            
            # Metadata statistics
            if split == 'train':
                print("\nTraining Set Metadata Statistics:")
                print(f"Total samples: {len(metadata_df)}")
                print("\nHearing Loss Distribution:")
                print(metadata_df['hearing_loss'].value_counts())
                print("\nCorrectness Statistics:")
                print(metadata_df['correctness'].describe())
                
                # Create intelligibility score distribution plot
                plt.figure(figsize=(10, 6))
                metadata_df['correctness'].hist(bins=20)
                plt.title('Distribution of Intelligibility Scores')
                plt.xlabel('Correctness Score')
                plt.ylabel('Count')
                plt.savefig(dataset_dir.parent / 'plots' / 'intelligibility_distribution.png')
                plt.close()
                
            else:  # Validation set
                print("\nValidation Set Metadata Statistics:")
                print(f"Total samples: {len(metadata_df)}")
                print("\nHearing Loss Distribution:")
                print(metadata_df['hearing_loss'].value_counts())
            
            # Store statistics
            stats[split] = {
                'n_signals': n_signals,
                'n_unprocessed': n_unprocessed,
                'metadata': metadata_df
            }
            
    return stats

def create_dataloaders(dataset_dir: Path, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders"""
    dataset_dir = Path(dataset_dir) / 'cadenza_data'
    
    train_dataset = CadenzaDataset(dataset_dir, split='train')
    valid_dataset = CadenzaDataset(dataset_dir, split='valid')
    
    logging.info(f"Created train dataset with {len(train_dataset)} samples")
    logging.info(f"Created validation dataset with {len(valid_dataset)} samples")
    
    # The ONLY change is adding the collate_fn argument here
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=pad_collate)
    
    return train_loader, valid_loader

def run_tests():
    """Run basic tests to verify dataset functionality"""
    current_dir = Path(__file__).parent
    data_dir = current_dir / "audio_files" / "16981327"
    
    # Test metadata loading
    metadata_path = data_dir / "cadenza_data" / "metadata" / "train_metadata.json"
    try:
        metadata_df = load_metadata(metadata_path)
        print("\nMetadata loading test passed!")
        print("\nSample metadata entry:")
        print(metadata_df.iloc[0])
    except Exception as e:
        print(f"Metadata loading test failed: {e}")
    
    # Test dataset creation
    try:
        dataset = CadenzaDataset(data_dir / "cadenza_data", split='train')
        print(f"\nDataset creation test passed!")
        print(f"Dataset size: {len(dataset)}")
        
        # Test single item loading
        item = dataset[0]
        print("\nSample item contents:")
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: Tensor of shape {value.shape}")
            elif isinstance(value, dict):
                print(f"{key}: {len(value)} metadata fields")
            else:
                print(f"{key}: {value}")
    except Exception as e:
        print(f"Dataset creation test failed: {e}")

if __name__ == "__main__":
    # Use absolute path and parent directory navigation
    current_dir = Path(__file__).parent
    data_dir = current_dir / "audio_files" / "16981327"
    
    # Print dataset statistics to verify file structure
    print("\n=== Dataset Statistics ===")
    stats = get_dataset_stats(data_dir)
    # Create dataloaders
    print("\n=== Creating DataLoaders ===")
    train_loader, valid_loader = create_dataloaders(data_dir, batch_size=32)
    print("\nDataloader created successfully!")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(valid_loader)}")
    
    # Run tests
    print("\n=== Running Tests ===")
    run_tests()



    # FOR TRAINING USE LIKE THIS 
    #     from data_loading import CadenzaDataset, create_dataloaders

    # # Initialize dataloaders
    # data_dir = Path("path/to/your/data")
    # train_loader, valid_loader = create_dataloaders(data_dir, batch_size=32)

    # # Use in training loop
    # for batch in train_loader:
    #     signals = batch['signal']  # Shape: [batch_size, channels, samples]
    #     unprocessed = batch['unprocessed']
    #     metadata = batch['metadata']  # Dictionary with all metadata fields
    #     # Your training code here