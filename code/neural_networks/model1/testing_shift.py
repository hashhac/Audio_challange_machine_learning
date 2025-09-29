import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import logging
import soundfile as sf
import torchaudio

# --- 1. SETUP: IMPORTS AND ROBUST PATHING ---
def setup_paths():
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent
    code_path = project_root / "code"
    if str(code_path) not in sys.path:
        sys.path.append(str(code_path))
    logging.info(f"Project root is: {project_root}")
    logging.info(f"Added to Python path: {code_path}")
    return project_root

project_root = setup_paths()

# Import the data loading utilities
from data_loading import create_dataloaders

# Import model components for reconstruction
try:
    from neural_networks.model_gen.model import (
        SpeechEncoder, AutoregressiveDecoder, ConditionalAudioConformerUNet,
        hearing_loss_to_tensor, autoregressive_generate, ddim_sampler
    )
except ImportError as e:
    logging.error(f"Failed to import model components: {e}")
    sys.exit(1)

# Import EnCodec for audio processing
try:
    from encodec import EncodecModel
    from encodec.utils import convert_audio
except ImportError:
    logging.error("EnCodec not available. Please install with: pip install -U encodec")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure hearing_loss_to_tensor is available locally if import fails
def hearing_loss_to_tensor_local(metadata_list, device=None):
    """Convert a list of metadata dicts (from CadenzaDataset) into a tensor conditioning vector.

    The mapping is simple and small by design so the model can condition on hearing loss type.
    Returns a float tensor of shape (batch,) with values in [0,1].
    Mapping:
      'No Loss' -> 0.0
      'Mild'    -> 0.33
      'Moderate'-> 0.66
      'Severe'  -> 1.0
    Unknown entries map to 0.0
    """
    mapping = {
        'No Loss': 0.0,
        'Mild': 0.33,
        'Moderate': 0.66,
        'Severe': 1.0
    }
    vals = []
    for md in metadata_list:
        try:
            hl = md.get('hearing_loss') if isinstance(md, dict) else None
            vals.append(mapping.get(hl, 0.0))
        except Exception:
            vals.append(0.0)
    tensor = torch.tensor(vals, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

# Use imported function if available, otherwise use local version
try:
    # Test if the imported function works
    test_metadata = [{'hearing_loss': 'Mild'}]
    hearing_loss_to_tensor(test_metadata)
    hearing_loss_fn = hearing_loss_to_tensor
    logging.info("Using imported hearing_loss_to_tensor function")
except:
    hearing_loss_fn = hearing_loss_to_tensor_local
    logging.info("Using local hearing_loss_to_tensor function")

# --- 2. NEURAL NETWORK MODEL CLASS ---
class S2S_Model:
    """
    Speech-to-Speech Neural Network Model for audio enhancement.
    Loads a trained model and provides audio processing capabilities.
    """
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Initialize the S2S model from a checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint (.pt file)
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device) if device else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        
        # Initialize EnCodec for audio tokenization/detokenization
        self.encodec_model = EncodecModel.encodec_model_24khz().to(self.device)
        try:
            self.encodec_model.set_target_bandwidth(12.0)  # High quality
        except Exception:
            self.encodec_model.set_target_bandwidth(6.0)   # Fallback
        self.encodec_model.eval()
        
        # Get vocab size from EnCodec
        self.vocab_size = self.encodec_model.quantizer.bins + 1
        self.mask_token_id = self.encodec_model.quantizer.bins
        
        # Model components (will be loaded from checkpoint)
        self.encoder = None
        self.ar_decoder = None
        self.diffusion_decoder = None
        
        # Hearing loss projection layer (same as in training)
        self.hearing_proj = torch.nn.Linear(1, 64).to(self.device)  # N_EMBED = 64
        
        # Load the trained model
        self._load_checkpoint()
        
        logging.info(f"S2S Model initialized on {self.device}")
        logging.info(f"Vocab size: {self.vocab_size}, Mask token: {self.mask_token_id}")

    def _load_checkpoint(self):
        """Load model components from checkpoint"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        logging.info(f"Loading checkpoint from: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Model hyperparameters (use defaults if not in checkpoint)
        n_embed = 64
        n_head = 4
        n_layer = 8
        block_size = 256
        dropout = 0.05
        
        # Initialize model components
        self.encoder = SpeechEncoder(
            vocab_sz=self.vocab_size,
            n_emb=n_embed,
            n_l=4,  # Encoder layers
            n_h=n_head,
            block_sz=block_size,
            dropout_val=dropout
        ).to(self.device)
        
        if 'ar_decoder_state_dict' in checkpoint:
            self.ar_decoder = AutoregressiveDecoder(
                vocab_sz=self.vocab_size,
                n_emb=n_embed,
                n_l=n_layer // 2,  # AR decoder layers
                n_h=n_head,
                block_sz=block_size,
                dropout_val=dropout
            ).to(self.device)
        
        if 'diffusion_decoder_state_dict' in checkpoint:
            self.diffusion_decoder = ConditionalAudioConformerUNet(
                vocab_sz=self.vocab_size,
                n_emb=n_embed,
                n_l=n_layer,  # Diffusion decoder layers
                n_h=n_head,
                block_sz=block_size,
                dropout_val=dropout
            ).to(self.device)
        
        # Load state dictionaries
        try:
            if 'encoder_state_dict' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
                logging.info("Loaded encoder weights")
            
            if self.ar_decoder and 'ar_decoder_state_dict' in checkpoint:
                self.ar_decoder.load_state_dict(checkpoint['ar_decoder_state_dict'], strict=False)
                logging.info("Loaded autoregressive decoder weights")
            
            if self.diffusion_decoder and 'diffusion_decoder_state_dict' in checkpoint:
                self.diffusion_decoder.load_state_dict(checkpoint['diffusion_decoder_state_dict'], strict=False)
                logging.info("Loaded diffusion decoder weights")
                
        except Exception as e:
            logging.error(f"Error loading model weights: {e}")
            raise
        
        # Set models to evaluation mode
        self.encoder.eval()
        if self.ar_decoder:
            self.ar_decoder.eval()
        if self.diffusion_decoder:
            self.diffusion_decoder.eval()
        
        logging.info("Model checkpoint loaded successfully")

    def _audio_to_tokens(self, audio_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Convert audio tensor to discrete tokens using EnCodec"""
        # Ensure audio is in correct format for EnCodec
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
        
        # Convert to target sample rate if needed
        if sample_rate != 24000:
            resampler = torchaudio.transforms.Resample(sample_rate, 24000)
            audio_tensor = resampler(audio_tensor)
        
        # Convert to EnCodec format
        audio_converted = convert_audio(
            audio_tensor, 24000, self.encodec_model.sample_rate, self.encodec_model.channels
        ).to(self.device)
        
        # Encode to tokens
        with torch.no_grad():
            encoded_frames = self.encodec_model.encode(audio_converted.unsqueeze(0))
            tokens = encoded_frames[0][0][0, 0, :].cpu()
        
        return tokens

    def _tokens_to_audio(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert discrete tokens back to audio using EnCodec"""
        # Prepare tokens for EnCodec decoder
        tokens_device = tokens.to(self.device)
        codes = torch.zeros(1, 1, len(tokens_device), dtype=torch.long, device=self.device)
        codes[0, 0, :] = torch.clamp(tokens_device, 0, self.vocab_size - 2)
        
        # Decode to audio
        with torch.no_grad():
            frames = [(codes, None)]
            audio = self.encodec_model.decode(frames)
            audio = audio.squeeze(0).cpu()  # Remove batch dimension
        
        return audio

    def process_audio(self, signal_tensor: torch.Tensor, sample_rate: int = 44100, 
                     metadata: dict = None) -> torch.Tensor:
        """
        Process audio using the trained S2S model with hearing loss conditioning.
        
        Args:
            signal_tensor: Input audio tensor of shape (channels, samples)
            sample_rate: Sample rate of input audio
            metadata: Metadata dictionary containing hearing loss information
            
        Returns:
            Processed audio tensor of same shape as input
        """
        with torch.no_grad():
            # Convert audio to tokens
            if signal_tensor.dim() == 2:
                # Process each channel separately and average the tokens
                channel_tokens = []
                for ch in range(signal_tensor.shape[0]):
                    tokens = self._audio_to_tokens(signal_tensor[ch:ch+1], sample_rate)
                    channel_tokens.append(tokens)
                # Use first channel tokens as primary, could average or combine differently
                source_tokens = channel_tokens[0]
            else:
                source_tokens = self._audio_to_tokens(signal_tensor, sample_rate)
            
            # Pad or truncate to model's expected length
            block_size = 256  # Model's block size
            if len(source_tokens) > block_size:
                source_tokens = source_tokens[:block_size]
            else:
                padding = torch.full((block_size - len(source_tokens),), 
                                   self.mask_token_id, dtype=torch.long)
                source_tokens = torch.cat([source_tokens, padding])
            
            # Add batch dimension
            source_tokens = source_tokens.unsqueeze(0).to(self.device)
            
            # Get conditioning vector from encoder
            conditioning_vector = self.encoder(source_tokens)
            
            # CRITICAL: Add hearing loss conditioning exactly like in training
            if metadata and 'hearing_loss' in metadata:
                # Use the same hearing_loss_to_tensor function from training
                hearing_tensor = hearing_loss_fn([metadata], device=self.device)
                # Project to embedding dimension and add to conditioning
                hearing_embed = self.hearing_proj(hearing_tensor.unsqueeze(-1))
                conditioning_vector = conditioning_vector + hearing_embed
                
                # Log the hearing loss being processed
                hearing_loss_label = metadata.get('hearing_loss', 'Unknown')
                hearing_value = hearing_tensor.item()
                logging.info(f"Applied hearing loss conditioning: {hearing_loss_label} -> {hearing_value:.2f}")
            else:
                # Default to "No Loss" if no metadata provided
                default_metadata = [{'hearing_loss': 'No Loss'}]
                hearing_tensor = hearing_loss_fn(default_metadata, device=self.device)
                hearing_embed = self.hearing_proj(hearing_tensor.unsqueeze(-1))
                conditioning_vector = conditioning_vector + hearing_embed
                logging.info("Applied default hearing loss conditioning: No Loss -> 0.0")
            
            # Generate enhanced tokens
            generated_tokens = None
            
            # Try autoregressive generation first
            if self.ar_decoder is not None:
                try:
                    generated_tokens = autoregressive_generate(
                        self.ar_decoder, conditioning_vector, max_length=64
                    )
                    logging.debug("Used autoregressive generation")
                except Exception as e:
                    logging.warning(f"Autoregressive generation failed: {e}")
            
            # Fallback to diffusion generation
            if generated_tokens is None and self.diffusion_decoder is not None:
                try:
                    generated_tokens = ddim_sampler(
                        self.diffusion_decoder, 
                        shape=(1, block_size), 
                        conditioning_vector=conditioning_vector,
                        num_steps=50,  # Fewer steps for faster inference
                        temperature=0.8
                    )
                    logging.debug("Used diffusion generation")
                except Exception as e:
                    logging.warning(f"Diffusion generation failed: {e}")
            
            # Final fallback: return processed version of input
            if generated_tokens is None:
                logging.warning("All generation methods failed, returning input")
                generated_tokens = source_tokens.cpu()
            
            # Convert tokens back to audio
            if generated_tokens.dim() == 2:
                generated_tokens = generated_tokens.squeeze(0)
            
            # Remove padding tokens
            valid_tokens = generated_tokens[generated_tokens != self.mask_token_id]
            if len(valid_tokens) == 0:
                valid_tokens = generated_tokens[:64]  # Use first 64 tokens if all are padding
            
            # Convert to audio
            enhanced_audio = self._tokens_to_audio(valid_tokens)
            
            # Match output channels to input
            if signal_tensor.dim() == 2 and signal_tensor.shape[0] > 1:
                # Duplicate mono output to match input channels
                if enhanced_audio.dim() == 1:
                    enhanced_audio = enhanced_audio.unsqueeze(0).repeat(signal_tensor.shape[0], 1)
                elif enhanced_audio.shape[0] == 1:
                    enhanced_audio = enhanced_audio.repeat(signal_tensor.shape[0], 1)
            
            # Ensure output length matches input (approximately)
            target_length = signal_tensor.shape[-1]
            current_length = enhanced_audio.shape[-1]
            
            if current_length > target_length:
                enhanced_audio = enhanced_audio[..., :target_length]
            elif current_length < target_length:
                padding_needed = target_length - current_length
                enhanced_audio = torch.nn.functional.pad(enhanced_audio, (0, padding_needed))
            
            return enhanced_audio


# --- 3. ENHANCED DATASET PROCESSING FUNCTION ---
def enhance_and_save_dataset(model: S2S_Model, dataloader: DataLoader, output_dir: Path, debug_pairs: int = 0):
    """
    Process dataset using the S2S neural network model with hearing loss conditioning.
    
    Args:
        model: Trained S2S model instance
        dataloader: DataLoader containing audio samples and metadata
        output_dir: Directory to save processed audio
        debug_pairs: Number of before/after pairs to save for comparison
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also save to s2s_audio_samples for consistency with training
    s2s_output_dir = output_dir.parent.parent / "s2s_audio_samples" / "testing_results"
    s2s_output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting S2S neural network enhancement process.")
    logging.info(f"Output will be saved to: {output_dir}")
    logging.info(f"S2S samples will also be saved to: {s2s_output_dir}")

    def normalize_and_clip(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
        """Normalize and clip audio to prevent artifacts"""
        x = x.astype('float32')
        max_abs = float(np.max(np.abs(x)) + 1e-12)
        if max_abs > 0:
            x = x / max_abs * peak
        np.clip(x, -1.0, 1.0, out=x)
        return x

    debug_count = 0
    hearing_loss_summary = {}

    for batch in tqdm(dataloader, desc="Processing Audio with S2S Model"):
        signals = batch['signal']
        metadata_list = batch['metadata']
        sample_rates = batch['sample_rate']

        for i in range(len(signals)):
            signal_id = None
            try:
                current_metadata = metadata_list[i]
                signal_id = current_metadata['signal']
                hearing_loss_label = current_metadata.get('hearing_loss', "No Loss")
                sample_rate = sample_rates[i] if isinstance(sample_rates, list) else sample_rates
                
                # Track hearing loss distribution
                if hearing_loss_label not in hearing_loss_summary:
                    hearing_loss_summary[hearing_loss_label] = 0
                hearing_loss_summary[hearing_loss_label] += 1
                
                logging.info(f"Processing {signal_id} with hearing loss: {hearing_loss_label}")

                # Process audio with S2S model (includes hearing loss conditioning)
                processed_tensor = model.process_audio(
                    signals[i], 
                    sample_rate=sample_rate, 
                    metadata=current_metadata
                )

                # Prepare original audio for saving
                orig_numpy = signals[i].cpu().numpy()
                if orig_numpy.ndim == 1:
                    orig_numpy = orig_numpy.reshape(1, -1)
                orig_to_write = orig_numpy.T.astype('float32')

                # Prepare processed audio for saving
                processed_numpy = processed_tensor.cpu().numpy()
                if processed_numpy.ndim == 1:
                    processed_numpy = processed_numpy.reshape(1, -1)
                proc_to_write = processed_numpy.T.astype('float32')

                # Apply safety normalization
                proc_to_write = normalize_and_clip(proc_to_write)

                # Save debug pairs (before/after comparison)
                if debug_count < debug_pairs:
                    idx = debug_count + 1
                    before_path = output_dir / f"{idx:02d}_before_{signal_id}_{hearing_loss_label}.wav"
                    after_path = output_dir / f"{idx:02d}_after_{signal_id}_{hearing_loss_label}.wav"
                    
                    # Also save to s2s_audio_samples
                    s2s_before_path = s2s_output_dir / f"source_{signal_id}_{hearing_loss_label}.wav"
                    s2s_after_path = s2s_output_dir / f"nn_processed_{signal_id}_{hearing_loss_label}.wav"
                    
                    logging.info(f"Saving debug pair {idx}: {before_path.name}, {after_path.name}")
                    
                    sf.write(before_path, orig_to_write, sample_rate)
                    sf.write(after_path, proc_to_write, sample_rate)
                    sf.write(s2s_before_path, orig_to_write, sample_rate)
                    sf.write(s2s_after_path, proc_to_write, sample_rate)
                    
                    debug_count += 1
                else:
                    # Save only processed output
                    output_filename = output_dir / f"{signal_id}_s2s_enhanced_{hearing_loss_label}.wav"
                    s2s_filename = s2s_output_dir / f"nn_processed_{signal_id}_{hearing_loss_label}.wav"
                    
                    sf.write(output_filename, proc_to_write, sample_rate)
                    sf.write(s2s_filename, proc_to_write, sample_rate)

            except Exception as e:
                logging.error(f"Error processing signal {signal_id or 'unknown'}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Print hearing loss distribution summary
    logging.info(f"S2S model processing complete. Check outputs in {output_dir}")
    logging.info("Hearing loss distribution in processed samples:")
    for hearing_loss, count in hearing_loss_summary.items():
        logging.info(f"  {hearing_loss}: {count} samples")


# --- 4. MAIN TESTING FUNCTION ---
def run_s2s_test(num_test_samples: int = 10, debug_pairs: int = 10):
    """
    Run S2S model test on a subset of validation data.
    
    Args:
        num_test_samples: Number of samples to process for testing
        debug_pairs: Number of before/after pairs to save
    """
    logging.info("--- Running S2S Neural Network Test ---")

    # Setup paths
    project_root = setup_paths()
    
    # Model checkpoint path
    checkpoint_path = project_root / "code" / "neural_networks" / "model1" / "model.pt"
    if not checkpoint_path.exists():
        logging.error(f"Model checkpoint not found: {checkpoint_path}")
        return
    
    # Output directory
    output_directory = project_root / "results" / "s2s_neural_network_test"

    # Data directory
    cadenza_data_root = project_root / "code" / "audio_files" / "16981327" / "cadenza_data"
    if not cadenza_data_root.exists():
        logging.error(f"Cadenza data not found: {cadenza_data_root}")
        return

    try:
        # Load the trained S2S model
        logging.info(f"Loading S2S model from: {checkpoint_path}")
        s2s_model = S2S_Model(checkpoint_path)
        
        # Create dataloader
        logging.info("Creating dataloaders...")
        _, full_valid_loader = create_dataloaders(cadenza_data_root.parent, batch_size=4)

        # Create test subset
        subset_indices = list(range(num_test_samples))
        test_dataset = Subset(full_valid_loader.dataset, subset_indices)
        test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=full_valid_loader.collate_fn)

        # Process the test data
        logging.info(f"Processing {num_test_samples} samples with S2S model...")
        enhance_and_save_dataset(s2s_model, test_loader, output_directory, debug_pairs=debug_pairs)
        
        print(f"\n=== S2S Neural Network Test Complete ===")
        print(f"Processed {num_test_samples} audio samples")
        print(f"Results saved to: {output_directory}")
        print(f"Debug pairs (before/after): {debug_pairs}")
        print(f"Enhanced audio files show the effect of your trained model!")
        print(f"Note: Files now include hearing loss labels in filenames for easy comparison")
        
    except Exception as e:
        logging.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Run the test with default parameters
    run_s2s_test(num_test_samples=10, debug_pairs=10)