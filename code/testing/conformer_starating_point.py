"""
Hybrid Speech-to-Speech (S2S) Model with Autoregressive + Diffusion Training

This implementation combines the best of both worlds:
1. **Autoregressive Decoder**: Learns proper sequential speech patterns through next-token prediction
2. **Diffusion Decoder**: Provides high-quality audio generation through denoising
3. **Long Sequences**: Supports ~30 seconds (1536 tokens) by concatenating random audio segments
4. **Chunked Training**: Processes long sequences in overlapping chunks for memory efficiency

Key Features:
- Hybrid training with configurable loss weights
- Conformer blocks for better speech modeling  
- Random sequence concatenation for diverse 30-second training samples
- Memory-efficient chunked autoregressive training
- Enhanced DDIM sampling with temperature annealing
- Comprehensive checkpointing for both decoders

Architecture:
- Encoder: 4-layer Conformer/Transformer → global conditioning vector
- AR Decoder: 4-layer causal Conformer/Transformer → sequential modeling
- Diffusion Decoder: 8-layer U-Net Conformer/Transformer → high-quality generation

The model learns both sequential patterns (via autoregressive) and high-quality synthesis (via diffusion),
making it capable of generating coherent, natural-sounding speech-to-speech conversions.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
import time
import os
import random
import math
import numpy as np
import traceback
# Set matplotlib backend to 'Agg' before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
# Import for gradient checkpointing
from torch.utils.checkpoint import checkpoint

# --- EnCodec Import ---
try:
    from encodec import EncodecModel
    from encodec.utils import convert_audio
except ImportError:
    print("Error: The 'encodec' package is not installed. Please install it using: pip install -U encodec")
    exit()

# --- Distributed Training Imports ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, DataLoader, Dataset

# Set environment variables for better compilation stability
os.environ['TORCHDYNAMO_VERBOSE'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '0'  # Disable debug for stability

# --- Advanced Feature Configuration ---
USE_CONFORMER = True               # Use Conformer blocks instead of Transformer blocks
USE_BITNET = False                 # Temporarily disable for stability with new architecture
USE_MOE = False                    # DISABLED - causes torch.compile graph fusion errors
USE_GRADIENT_CHECKPOINTING = False # DISABLED - conflicts with torch.compile
USE_TORCH_COMPILE = True           # DISABLED - causes CUDA graph tensor overwriting issues
USE_STOCHASTIC_SAMPLING = True    # Use deterministic sampling for cleaner results
NUM_EXPERTS = 4                    # Number of experts for the MoE layer
DDIM_SAMPLING_STEPS = 100          # More steps for cleaner sampling
SAMPLING_TEMPERATURE = 0.9        # Temperature for sampling control
CLEAN_SAMPLING_MODE = True         # Enable all clean sampling optimizations
CLEAR_CACHE = True                  # Clear CUDA cache periodically for stability only if small memory issues
# unhappy compiler stuff
cl = torch._C._get_tracing_state()

# --- S2S Configuration ---
DATASET_PATH = Path(os.path.join("DS_10283_3443", "VCTK-Corpus-0.92")) # <-- IMPORTANT: Set this to your VCTK dataset path
TARGET_SAMPLE_RATE = 24000 # EnCodec's native sample rate

# --- EnCodec Quality Settings ---
ENCODEC_BANDWIDTH = 12.0  # Increase from 6.0 to 12.0 for much better quality (2x bitrate)
# Available bandwidths: 1.5, 3.0, 6.0, 12.0, 24.0 kbps
# 12.0 kbps provides excellent quality while still being efficient
# 24.0 kbps is near-lossless but uses more memory/compute

# Quality presets for easy experimentation:
# ENCODEC_BANDWIDTH = 6.0   # Default - fast training, decent quality
# ENCODEC_BANDWIDTH = 12.0  # High quality - 2x better audio, slightly slower
# ENCODEC_BANDWIDTH = 24.0  # Near-lossless - best quality, uses more resources

# Audio processing quality settings
USE_AUDIO_ENHANCEMENT = True  # Enable audio preprocessing and post-processing
AUDIO_NORMALIZATION = True    # Normalize audio levels for consistency

# --- Base Configuration ---
BATCH_SIZE = 16 # Reduced batch size for faster iterations and more gradient updates
BLOCK_SIZE = 512 # Shorter sequences = faster training, still ~6-8 seconds of audio
N_EMBED = 64#320 # Slightly larger embedding for better capacity
N_LAYER = 8   # Increase layers for better model capacity (should be even for U-Net)
N_HEAD = 4 # More attention heads for better representation
MAX_ITERS = 25001 # More iterations to compensate for smaller batches
CHECKPOINT_INTERVAL = 1000 # Save checkpoints much more frequently to avoid loss
EVAL_INTERVAL = 100 # More frequent evaluation to monitor progress
LEARNING_RATE = 0.24/N_EMBED # Higher fixed learning rate for faster convergence
MIN_LEARNING_RATE = 0.0666/N_EMBED  # Minimum learning rate for cosine decay
WARMUP_STEPS = 500  # Learning rate warmup
DROPOUT = 0.05 # Slightly higher dropout to prevent overfitting
COMPILE_MODE = "default" # Faster compilation mode

# Training optimizations
USE_MIXED_PRECISION = True  # Use automatic mixed precision for speed
GRADIENT_CLIP_VAL = 1.0  # Gradient clipping for stability

# Loss monitoring for training insights
LOSS_HISTORY_SIZE = 100  # Track recent losses for averaging
loss_history = {
    'total': [],
    'autoregressive': [],
    'diffusion': []
}
# --- Hybrid Training Configuration ---
USE_HYBRID_TRAINING = True    # Enable hybrid autoregressive + diffusion training
AUTOREGRESSIVE_WEIGHT = 0.7   # Slightly favor autoregressive for coherence
DIFFUSION_WEIGHT = 0.4        # Reduce diffusion weight to speed up training
AR_CHUNK_SIZE = 64            # Smaller chunks for faster processing
SEQUENCE_OVERLAP = 32         # Reduce overlap for speed
CHECKPOINT_DIR = Path("s2s_conformer_checkpoints")
LOG_DIR = Path("runs/s2s_conformer_diffusion")
PLOT_DIR = Path("plots/s2s_conformer_diffusion")
DATA_CACHE_PATH = Path("speech_commands_tokenized.pt")
PROGRESS_CACHE_PATH = Path("tokenization_progress.pt")
AUDIO_SAMPLES_DIR = Path("s2s_audio_samples")

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(AUDIO_SAMPLES_DIR, exist_ok=True)

# Diffusion Hyperparameters
NUM_TIMESTEPS = 200

# DDP Setup
DDP_BACKEND = 'nccl'
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
RANK = int(os.environ.get('RANK', 0))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
DEVICE = torch.device(f"cuda:{LOCAL_RANK}" if WORLD_SIZE > 1 and torch.cuda.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

# --- DDP and Seeding Functions ---
def setup_ddp():
    if WORLD_SIZE > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend=DDP_BACKEND)
        torch.cuda.set_device(LOCAL_RANK)

def cleanup_ddp():
    if WORLD_SIZE > 1 and dist.is_initialized():
        dist.destroy_process_group()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

seed_everything()

# --- Data Loading and Preprocessing (Unchanged) ---
full_data_tensor = None
vocab_size = -1
MASK_TOKEN_ID = -1
encodec_model = None

def load_and_prep_data():
    global full_data_tensor, vocab_size, MASK_TOKEN_ID, encodec_model
    encodec_model = EncodecModel.encodec_model_24khz().to(DEVICE)
    encodec_model.set_target_bandwidth(6.0) # question also base niose might be needed for smoother audio transitionsa and long term fidelity audio segments without drift
    encodec_model.eval()
    codebook_size = encodec_model.quantizer.bins
    vocab_size = codebook_size + 1
    MASK_TOKEN_ID = codebook_size

    if RANK == 0:
        print(f"EnCodec model loaded. Vocab size: {codebook_size}, MASK_TOKEN_ID: {MASK_TOKEN_ID}")

    if RANK == 0 and not DATA_CACHE_PATH.exists():
        print(f"Tokenizing dataset, caching to {DATA_CACHE_PATH}")
        dataset = VCTKPairDataset(DATASET_PATH, BLOCK_SIZE)  # Use the new VCTK dataset handler  WILL NEED TO FIX AND MAKE SURE WITH TESTS AND EXPLORATION OF THIS DATA-set 
        # (Tokenization logic remains the same as in your original code) This is by far the biggest area for improvement in terms of speed and memory usage it can be so much better
        tokenized_data = []
        for i in tqdm(range(len(dataset)), desc="Tokenizing audio"):
            waveform, sample_rate, _, _, _ = dataset[i]
            waveform_converted = convert_audio(waveform, sample_rate, encodec_model.sample_rate, encodec_model.channels).to(DEVICE)
            with torch.no_grad():
                encoded_frames = encodec_model.encode(waveform_converted.unsqueeze(0))
                codes = encoded_frames[0][0]
                tokens = codes[0, 0, :].cpu()
            if tokens.size(0) > BLOCK_SIZE:
                tokens = tokens[:BLOCK_SIZE]
            else:
                padding = torch.full((BLOCK_SIZE - tokens.size(0),), MASK_TOKEN_ID, dtype=torch.long)
                tokens = torch.cat([tokens, padding])
            tokenized_data.append(tokens)
        full_data_tensor = torch.stack(tokenized_data)
        torch.save(full_data_tensor, DATA_CACHE_PATH)
    
    if WORLD_SIZE > 1: dist.barrier()
    if RANK == 0: print("Loading tokenized data from cache...")
    full_data_tensor = torch.load(DATA_CACHE_PATH, map_location='cpu')
    if RANK == 0: print(f"Data loaded. Tensor shape: {full_data_tensor.shape}")

# --- NEW: VCTK Dataset Handler ---
class VCTKPairDataset(Dataset):
    def __init__(self, root_path, block_size):
        super().__init__()
        self.root_path = Path(root_path)
        self.block_size = block_size
        self._encodec_model = None  # Lazy initialization
        
        # Simply collect all available audio files
        self.audio_files = []
        
        # Try multiple possible patterns for audio files
        patterns = [
            'wav48_silence_trimmed/**/*.flac',
            'wav48/**/*.flac', 
            '**/*.wav',
            '**/*.flac',
            'wav/**/*.wav'
        ]
        
        for pattern in patterns:
            files = list(self.root_path.glob(pattern))
            if files:
                self.audio_files.extend(files)
                if RANK == 0:
                    print(f"Found {len(files)} files with pattern: {pattern}")
                break
        
        if not self.audio_files:
            if RANK == 0:
                print(f"No audio files found in {self.root_path}")
                print("Checked patterns:", patterns)
        
        if RANK == 0:
            print(f"Total audio files found: {len(self.audio_files)}")
            if self.audio_files:
                print(f"Sample files: {[f.name for f in self.audio_files[:5]]}")
                print(f"File structure example: {self.audio_files[0]}")

    @property
    def encodec_model(self):
        """Lazy initialization of encodec model for worker processes"""
        if self._encodec_model is None:
            self._encodec_model = EncodecModel.encodec_model_24khz()
            self._encodec_model.set_target_bandwidth(6.0)
            self._encodec_model.eval()
            # Move to appropriate device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._encodec_model = self._encodec_model.to(device)
        return self._encodec_model

    def __len__(self):
        # Use all available audio files
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Enhanced strategy: Create 2-segment combinations for more diversity
        # This doubles our effective dataset size while keeping manageable sequence lengths
        
        # Select 2 random audio files for concatenation
        file1_idx = random.randint(0, len(self.audio_files) - 1)
        file2_idx = random.randint(0, len(self.audio_files) - 1)
        
        # Ensure different files for more diversity
        while file2_idx == file1_idx and len(self.audio_files) > 1:
            file2_idx = random.randint(0, len(self.audio_files) - 1)
        
        # Load and tokenize both files
        tokens1 = self._load_and_tokenize_raw(self.audio_files[file1_idx])
        tokens2 = self._load_and_tokenize_raw(self.audio_files[file2_idx])
        
        # Remove padding tokens for clean concatenation
        mask_token_id = self.encodec_model.quantizer.bins
        valid_tokens1 = tokens1[tokens1 != mask_token_id]
        valid_tokens2 = tokens2[tokens2 != mask_token_id]
        
        # Create sequence by combining both
        sequence_tokens = []
        
        # Add first segment (up to half block size)
        if len(valid_tokens1) > 0:
            max_len1 = min(len(valid_tokens1), self.block_size // 2)
            start1 = random.randint(0, max(0, len(valid_tokens1) - max_len1))
            sequence_tokens.extend(valid_tokens1[start1:start1 + max_len1].tolist())
        
        # Add second segment to fill remaining space
        remaining_space = self.block_size - len(sequence_tokens)
        if len(valid_tokens2) > 0 and remaining_space > 0:
            max_len2 = min(len(valid_tokens2), remaining_space)
            start2 = random.randint(0, max(0, len(valid_tokens2) - max_len2))
            sequence_tokens.extend(valid_tokens2[start2:start2 + max_len2].tolist())
        
        # Pad to exact block size if needed
        while len(sequence_tokens) < self.block_size:
            sequence_tokens.append(mask_token_id)
        
        # Truncate if too long
        sequence_tokens = sequence_tokens[:self.block_size]
        
        sequence_tensor = torch.tensor(sequence_tokens, dtype=torch.long)
        
        # For S2S, use same sequence as both source and target
        return sequence_tensor, sequence_tensor

    def _load_and_tokenize_raw(self, file_path):
        """Load and tokenize without padding - for flexible concatenation"""
        model = self.encodec_model
        device = next(model.parameters()).device
        
        try:
            waveform, sr = torchaudio.load(file_path)
            
            # Audio preprocessing for better quality (conditional)
            if USE_AUDIO_ENHANCEMENT:
                # 1. Ensure mono audio
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # 2. Normalize audio to prevent clipping
                if AUDIO_NORMALIZATION:
                    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
                
                # 3. Apply gentle high-pass filter to remove DC offset and low-freq noise
                if sr != TARGET_SAMPLE_RATE:
                    # Resample with high-quality resampling
                    resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE, resampling_method="sinc_interp_kaiser")
                    waveform = resampler(waveform)
            
            waveform_converted = convert_audio(waveform, sr if not USE_AUDIO_ENHANCEMENT else TARGET_SAMPLE_RATE, TARGET_SAMPLE_RATE, model.channels).to(device)

            # 4. Ensure minimum length for stable encoding
            if USE_AUDIO_ENHANCEMENT:
                min_length = int(0.1 * TARGET_SAMPLE_RATE)  # 100ms minimum
                if waveform_converted.shape[-1] < min_length:
                    padding_needed = min_length - waveform_converted.shape[-1]
                    waveform_converted = F.pad(waveform_converted, (0, padding_needed))

            with torch.no_grad():
                encoded_frames = model.encode(waveform_converted.unsqueeze(0))
                tokens = encoded_frames[0][0][0, 0, :].cpu() # Shape: (T)
            
            return tokens
        except Exception as e:
            if RANK == 0:
                print(f"Warning: Could not load {file_path}: {e}")
            # Return small valid sequence on error
            return torch.randint(0, model.quantizer.bins - 1, (50,), dtype=torch.long)

    def _load_and_tokenize(self, file_path):
        # Use the lazy-loaded encodec model
        model = self.encodec_model
        device = next(model.parameters()).device
        
        waveform, sr = torchaudio.load(file_path)
        waveform_converted = convert_audio(waveform, sr, TARGET_SAMPLE_RATE, model.channels).to(device)

        with torch.no_grad():
            encoded_frames = model.encode(waveform_converted.unsqueeze(0))
            tokens = encoded_frames[0][0][0, 0, :].cpu() # Shape: (T)
        
        # Pad or truncate to block_size
        if tokens.size(0) > self.block_size:
            start = random.randint(0, tokens.size(0) - self.block_size - 1)
            tokens = tokens[start : start + self.block_size]
        else:
            # Use the model's vocab size for MASK_TOKEN_ID
            mask_token_id = model.quantizer.bins
            padding = torch.full((self.block_size - tokens.size(0),), mask_token_id, dtype=torch.long)
            tokens = torch.cat([tokens, padding])
        
        return tokens

# --- Diffusion and Model Components ---

# --- BitNet b1.58 Layer ---
class BitLinear1_58(nn.Linear):
    """
    A 1.58-bit Linear layer that quantizes weights to {-1, 0, 1}.
    """
    def __init__(self, in_features, out_features, bias=True, quantization_threshold_factor=0.1):
        super().__init__(in_features, out_features, bias) # Initialize the parent class good job <3
        self.quantization_threshold_factor = quantization_threshold_factor

    def forward(self, x):
        # Weight quantization during both training and inference
        beta = self.weight.abs().mean() + 1e-9  # Group scaling factor
        centered_w = self.weight - self.weight.mean() # is this correct? I think it is mmm something for me to think on i guess
        
        std_dev = centered_w.std() + 1e-9
        threshold_val = self.quantization_threshold_factor * std_dev
        
        w_quant_ternary = torch.zeros_like(centered_w)
        w_quant_ternary[centered_w > threshold_val] = 1.0
        w_quant_ternary[centered_w < -threshold_val] = -1.0
        
        w_effective = w_quant_ternary * beta

        # Straight-Through Estimator (STE)
        quantized_weight_for_forward = self.weight - self.weight.detach() + w_effective.detach()

        # No activation quantization in this version for simplicity
        return F.linear(x, quantized_weight_for_forward, self.bias)

# Helper function to select Linear layer type
def get_linear_layer(in_features, out_features, bias=True):
    if USE_BITNET:
        return BitLinear1_58(in_features, out_features, bias)
    else:
        return nn.Linear(in_features, out_features, bias)

# --- MoE Layer ---
class MoEPositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff_multiplier=4, dropout=DROPOUT):
        super().__init__()
        d_ff = d_model * d_ff_multiplier
        self.fc1 = get_linear_layer(d_model, d_ff, True)
        self.fc2 = get_linear_layer(d_ff, d_model, True)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class MoELayer(nn.Module):
    def __init__(self, n_embed, num_experts=NUM_EXPERTS, dropout=DROPOUT):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([MoEPositionwiseFFN(n_embed, dropout=dropout) for _ in range(num_experts)])
        self.gate = get_linear_layer(n_embed, num_experts, bias=False)

    def forward(self, x):
        gate_logits = self.gate(x)
        
        # Use tanh-based normalization instead of softmax for better compiler compatibility
        # This creates a smoother, more stable gating mechanism
        gate_tanh = torch.tanh(gate_logits)
        gate_weights = F.normalize(gate_tanh.abs() + 1e-8, p=1, dim=-1)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        moe_output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=2)
        return moe_output

# --- Conformer Block Components ---
class ConvolutionModule(nn.Module):
    def __init__(self, channels, kernel_size=31):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, groups=channels, padding='same')
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x):
        x_res = x
        x = x.permute(0, 2, 1) # (B, T, C) -> (B, C, T)
        x, gate = self.pointwise_conv1(x).chunk(2, dim=1)
        x = self.activation(x) * torch.sigmoid(gate)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv2(x)
        x = x.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
        return self.layer_norm(x + x_res)

class TransformerBlock(nn.Module):
    """Fallback Transformer block when Conformer is disabled"""
    def __init__(self, n_embed, n_head, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embed, n_head, dropout=dropout, batch_first=True)
        self.ffn = MoELayer(n_embed, num_experts=NUM_EXPERTS, dropout=dropout) if USE_MOE else MoEPositionwiseFFN(n_embed, dropout=dropout)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class ConformerBlock(nn.Module):
    def __init__(self, n_embed, n_head, dropout):
        super().__init__()
        self.ffn1 = MoEPositionwiseFFN(n_embed, d_ff_multiplier=2, dropout=dropout) # Macaron FFN
        self.attn = nn.MultiheadAttention(n_embed, n_head, dropout=dropout, batch_first=True)
        self.conv = ConvolutionModule(n_embed)
        self.ffn2 = MoELayer(n_embed, num_experts=NUM_EXPERTS, dropout=dropout) if USE_MOE else MoEPositionwiseFFN(n_embed, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)
        self.norm3 = nn.LayerNorm(n_embed)
        self.norm4 = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + 0.5 * self.dropout(self.ffn1(self.norm1(x)))
        x = x + self.dropout(self.attn(self.norm2(x), self.norm2(x), self.norm2(x))[0])
        x = x + self.dropout(self.conv(self.norm3(x)))
        x = x + 0.5 * self.dropout(self.ffn2(self.norm4(x)))
        return x

# --- NEW: Autoregressive Decoder for Hybrid Training ---
class AutoregressiveDecoder(nn.Module):
    def __init__(self, vocab_sz, n_emb, n_l, n_h, block_sz, dropout_val):
        super().__init__()
        Block = ConformerBlock if USE_CONFORMER else TransformerBlock
        
        self.token_embedding = nn.Embedding(vocab_sz, n_emb)
        self.positional_encoding = PositionalEncoding(n_emb, dropout_val, max_len=block_sz)
        
        # Autoregressive transformer layers with causal attention
        self.layers = nn.ModuleList([Block(n_emb, n_h, dropout_val) for _ in range(n_l//2)])  # Smaller than diffusion decoder
        self.ln_f = nn.LayerNorm(n_emb)
        self.output_projection = get_linear_layer(n_emb, vocab_sz - 1, bias=True)
        
        # Conditioning projection for source speech
        self.conditioning_projection = nn.Linear(n_emb, n_emb)
        
        if RANK == 0: 
            print("Initialized Autoregressive Decoder for sequential modeling.")

    def forward(self, target_tokens, conditioning_vector, return_logits_for_all_positions=True):
        B, T = target_tokens.shape
        
        # Token and position embeddings
        token_embed = self.token_embedding(target_tokens)
        pos_embed = self.positional_encoding(token_embed)
        
        # Add conditioning vector to each position
        cond_embed = self.conditioning_projection(conditioning_vector).unsqueeze(1).repeat(1, T, 1)
        x = pos_embed + cond_embed
        
        # Apply transformer layers with causal masking
        for layer in self.layers:
            if hasattr(layer, 'attn'):  # For both ConformerBlock and TransformerBlock
                # Create causal mask for autoregressive generation
                causal_mask = torch.triu(torch.ones(T, T, device=target_tokens.device), diagonal=1).bool()
                
                if isinstance(layer, ConformerBlock):
                    # For Conformer, we need to manually apply causal masking in attention
                    x_norm = layer.norm2(x)
                    attn_out, _ = layer.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask)
                    x = x + layer.dropout(attn_out)
                    
                    # Apply other Conformer components normally
                    x = x + 0.5 * layer.dropout(layer.ffn1(layer.norm1(x)))
                    x = x + layer.dropout(layer.conv(layer.norm3(x)))
                    x = x + 0.5 * layer.dropout(layer.ffn2(layer.norm4(x)))
                else:
                    # For TransformerBlock, apply causal masking
                    x_norm = layer.norm1(x)
                    attn_out, _ = layer.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask)
                    x = x + layer.dropout(attn_out)
                    x = x + layer.dropout(layer.ffn(layer.norm2(x)))
            else:
                # Fallback for layers without attention
                if self.training and USE_GRADIENT_CHECKPOINTING:
                    x = checkpoint(layer, x, use_reentrant=False)
                else:
                    x = layer(x)
        
        x = self.ln_f(x)
        logits = self.output_projection(x)
        
        if return_logits_for_all_positions:
            return logits
        else:
            return logits[:, -1, :]  # Only return last position for generation

# --- NEW: Speech Encoder ---
class SpeechEncoder(nn.Module):
    def __init__(self, vocab_sz, n_emb, n_l, n_h, block_sz, dropout_val):
        super().__init__()
        Block = ConformerBlock if USE_CONFORMER else TransformerBlock
        self.token_embedding = nn.Embedding(vocab_sz, n_emb)
        self.positional_encoding = PositionalEncoding(n_emb, dropout_val, max_len=block_sz)
        
        self.layers = nn.ModuleList([Block(n_emb, n_h, dropout_val) for _ in range(n_l)])
        self.ln_f = nn.LayerNorm(n_emb)
        # Global average pooling to get a single conditioning vector
        self.pool = nn.AdaptiveAvgPool1d(1)

        if RANK == 0: print("Initialized Speech Encoder.")

    def forward(self, source_tokens):
        x = self.token_embedding(source_tokens)
        x = self.positional_encoding(x)
        
        for layer in self.layers:
            if self.training and USE_GRADIENT_CHECKPOINTING:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        
        x = self.ln_f(x)
        # Clone tensor before pooling to prevent CUDA graph overwriting issues
        x_cloned = x.clone() if hasattr(x, 'clone') else x
        x = self.pool(x_cloned.permute(0, 2, 1)).squeeze(-1) # (B, C, T) -> (B, C, 1) -> (B, C)
        return x

# --- Main U-Net Model ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb_const = math.log(10000) / (half_dim - 1 if half_dim > 1 else 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -emb_const)
        embeddings = time.float().unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings

class ConditionalAudioConformerUNet(nn.Module):
    def __init__(self, vocab_sz, n_emb, n_l, n_h, block_sz, dropout_val):
        super().__init__()
        assert n_l % 2 == 0
        self.n_levels = n_l // 2
        Block = ConformerBlock if USE_CONFORMER else TransformerBlock # TransformerBlock for fallback

        self.token_embedding = nn.Embedding(vocab_sz, n_emb)
        self.time_embedding = SinusoidalTimeEmbedding(n_emb)
        self.positional_encoding = PositionalEncoding(n_emb, dropout_val, max_len=block_sz)
        
        self.down_blocks = nn.ModuleList([Block(n_emb, n_h, dropout_val) for _ in range(self.n_levels)])
        self.pools = nn.ModuleList([nn.AvgPool1d(kernel_size=2, stride=2) for _ in range(self.n_levels)])
        
        self.mid_block = Block(n_emb, n_h, dropout_val)
        
        self.up_samples = nn.ModuleList([nn.Upsample(scale_factor=2, mode='nearest') for _ in range(self.n_levels)])
        self.up_blocks = nn.ModuleList([Block(n_emb, n_h, dropout_val) for _ in range(self.n_levels)])
        self.up_projs = nn.ModuleList([get_linear_layer(n_emb * 2, n_emb) for _ in range(self.n_levels)])
        
        self.ln_f = nn.LayerNorm(n_emb)
        self.output_projection = get_linear_layer(n_emb, vocab_sz - 1, bias=True)
        
        # NEW: Projection for the conditioning vector
        self.conditioning_projection = nn.Linear(n_emb, n_emb) # mm was this needed?

        if RANK == 0:
            print("--- Conditional AudioConformerUNet Architecture ---")
            print(f"Using Conformer: {USE_CONFORMER}, BitNet: {USE_BITNET}, MoE: {USE_MOE} ({NUM_EXPERTS} experts)")
            print(f"Layers: {n_l} total, Embed Dim: {n_emb}, Heads: {n_h}")
            print(f"Gradient Checkpointing: {USE_GRADIENT_CHECKPOINTING}")

    def forward(self, x_t_tokens, t, conditioning_vector): # <-- NEW argument
        B, T = x_t_tokens.shape
        token_embed = self.token_embedding(x_t_tokens)
        pos_embed = self.positional_encoding(token_embed)
        time_embed = self.time_embedding(t).unsqueeze(1)
        
        # NEW: Project and add the conditioning vector to the time embedding
        cond_embed = self.conditioning_projection(conditioning_vector).unsqueeze(1)
        combined_embed = (time_embed + cond_embed).repeat(1, T, 1)

        x = pos_embed + combined_embed
        
        skip_connections = []
        for i in range(self.n_levels):
            if self.training and USE_GRADIENT_CHECKPOINTING:
                x = checkpoint(self.down_blocks[i], x, use_reentrant=False)
            else:
                x = self.down_blocks[i](x)
            skip_connections.append(x)
            x = self.pools[i](x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.training and USE_GRADIENT_CHECKPOINTING:
            x = checkpoint(self.mid_block, x, use_reentrant=False)
        else:
            x = self.mid_block(x)

        for i in range(self.n_levels):
            skip = skip_connections.pop()
            x = self.up_samples[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x = torch.cat([x, skip], dim=-1)
            x = self.up_projs[i](x)
            if self.training and USE_GRADIENT_CHECKPOINTING:
                 x = checkpoint(self.up_blocks[i], x, use_reentrant=False)
            else:
                 x = self.up_blocks[i](x)
            
        return self.output_projection(self.ln_f(x))

# --- Hybrid Training Functions ---
def compute_autoregressive_loss(ar_decoder, source_tokens, target_tokens, conditioning_vector):
    """Compute autoregressive loss for sequential modeling"""
    B, T = target_tokens.shape
    
    # For autoregressive training, we predict the next token given previous tokens
    # Input: tokens[:-1], Target: tokens[1:]
    input_tokens = target_tokens[:, :-1]  # All tokens except the last
    target_next_tokens = target_tokens[:, 1:]  # All tokens except the first
    
    # Get predictions for all positions
    logits = ar_decoder(input_tokens, conditioning_vector, return_logits_for_all_positions=True)
    
    # Compute cross-entropy loss for next token prediction
    ar_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), 
        target_next_tokens.reshape(-1), 
        ignore_index=MASK_TOKEN_ID
    )
    
    return ar_loss

def compute_chunked_autoregressive_loss(ar_decoder, source_tokens, target_tokens, conditioning_vector, chunk_size=AR_CHUNK_SIZE):
    """Compute autoregressive loss on chunks for memory efficiency with longer sequences"""
    B, T = target_tokens.shape
    total_loss = 0.0
    num_chunks = 0
    
    # Process sequence in overlapping chunks
    for start_pos in range(0, T - chunk_size, chunk_size - SEQUENCE_OVERLAP):
        end_pos = min(start_pos + chunk_size, T)
        
        # Extract chunk
        chunk_tokens = target_tokens[:, start_pos:end_pos]
        
        # Skip if chunk is too small
        if chunk_tokens.size(1) < 2:
            continue
            
        # Compute autoregressive loss for this chunk
        chunk_loss = compute_autoregressive_loss(ar_decoder, source_tokens, chunk_tokens, conditioning_vector)
        total_loss += chunk_loss
        num_chunks += 1
    
    return total_loss / max(num_chunks, 1)

def compute_diffusion_loss(diffusion_decoder, source_tokens, target_tokens, conditioning_vector):
    """Compute diffusion loss for high-quality generation"""
    t = torch.randint(0, NUM_TIMESTEPS, (target_tokens.shape[0],), device=target_tokens.device).long()
    x_t = q_sample_discrete(target_tokens, t, mask_token_id=MASK_TOKEN_ID)
    
    predicted_x0_logits = diffusion_decoder(x_t, t, conditioning_vector)
    
    diffusion_loss = F.cross_entropy(
        predicted_x0_logits.reshape(-1, predicted_x0_logits.size(-1)), 
        target_tokens.reshape(-1), 
        ignore_index=MASK_TOKEN_ID
    )
    
    return diffusion_loss

# --- Diffusion Process & DDIM Sampler ---
def get_corruption_probability(t, num_timesteps=NUM_TIMESTEPS):
    return (t.float() + 1) / num_timesteps

def q_sample_discrete(x_start, t, mask_token_id):
    t_dev = t.to(x_start.device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=x_start.device)
    if t_dev.ndim == 0: t_dev = t_dev.repeat(x_start.shape[0])
    corruption_probs = get_corruption_probability(t_dev).unsqueeze(1).repeat(1, x_start.shape[1])
    should_corrupt = torch.rand_like(corruption_probs, device=x_start.device) < corruption_probs
    mask_tensor = torch.full_like(x_start, mask_token_id, device=x_start.device, dtype=x_start.dtype)
    
    # Ensure x_start is in valid range before processing
    x_start_clamped = torch.clamp(x_start, 0, mask_token_id - 1)
    
    return torch.where(should_corrupt, mask_tensor, x_start_clamped)

@torch.no_grad()
def autoregressive_generate(ar_model, conditioning_vector, max_length=None, temperature=0.7, start_tokens=None):
    """Generate speech autoregressively for comparison/fallback"""
    if max_length is None:
        max_length = BLOCK_SIZE // 4  # Shorter for autoregressive to save time
    
    ar_model.eval()
    device = next(ar_model.parameters()).device
    
    if start_tokens is None:
        # Start with a few random valid tokens
        generated = torch.randint(0, vocab_size - 2, (1, 4), device=device)
    else:
        generated = start_tokens.clone()
    
    for _ in range(max_length - generated.size(1)):
        # Get prediction for next token
        logits = ar_model(generated, conditioning_vector, return_logits_for_all_positions=False)
        
        # Apply temperature and sample
        logits = logits / temperature
        next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
        
        # Append to sequence
        generated = torch.cat([generated, next_token], dim=1)
        
        # Stop if we hit padding token
        if next_token.item() == MASK_TOKEN_ID:
            break
    
    ar_model.train()
    return generated.cpu()

@torch.no_grad()
def ddim_sampler(model, shape, conditioning_vector=None, num_steps=DDIM_SAMPLING_STEPS, temperature=SAMPLING_TEMPERATURE):
    """
    DDIM sampler for discrete data.
    This is a deterministic sampler (eta=0) that uses the model's prediction of x0
    to iteratively denoise from a masked state to a clean signal.
    """
    global vocab_size, MASK_TOKEN_ID
    model.eval()
    B, T = shape
    
    # Start from a fully masked tensor
    x_t = torch.full(shape, MASK_TOKEN_ID, dtype=torch.long, device=DEVICE)
    
    # Define the timesteps to sample at, from T-1 down to 0
    times = torch.linspace(NUM_TIMESTEPS - 1, 0, steps=num_steps).long()
    
    for i in tqdm(range(num_steps), desc="DDIM Sampling", leave=False):
        t = times[i]
        
        # The time for the next step in the sequence
        t_prev = times[i+1] if i < num_steps - 1 else -1

        t_tensor = torch.full((B,), t, dtype=torch.long, device=DEVICE)
        
        # 1. Predict the original signal (x0) from the noisy input x_t
        pred_x0_logits = model(x_t, t_tensor, conditioning_vector)
        
        # Apply temperature and get the most likely x0 prediction
        pred_x0_logits = pred_x0_logits / temperature
        pred_x0 = torch.argmax(pred_x0_logits, dim=-1)
        pred_x0 = torch.clamp(pred_x0, 0, vocab_size - 2) # Ensure validity
        
        # If this is the last step, the prediction of x0 is our final result
        if t_prev < 0:
            x_t = pred_x0
            break
            
        # 2. Re-noise the predicted x0 to the level of the next step (t_prev)
        # This is the core of the deterministic DDIM step for discrete data.
        # We use the forward process `q_sample_discrete` to move from our x0 estimate
        # to the corresponding noisy sample at the previous timestep.
        x_t = q_sample_discrete(pred_x0, torch.tensor([t_prev], device=DEVICE), MASK_TOKEN_ID)

    model.train()
    
    # Final quality check and cleanup
    x_t = torch.clamp(x_t, 0, vocab_size - 2)
    
    # Replace any remaining mask tokens
    mask_positions = (x_t == MASK_TOKEN_ID)
    if mask_positions.any():
        valid_tokens = x_t[x_t != MASK_TOKEN_ID]
        if len(valid_tokens) > 0:
            replacement_token = torch.mode(valid_tokens).values.item()
        else:
            replacement_token = vocab_size // 2
        x_t[mask_positions] = replacement_token
        
    return x_t.cpu()

# --- Main Script Execution ---
def main():
    setup_ddp()
    if RANK == 0:
        if torch.cuda.is_available() and torch.cuda.get_device_capability(DEVICE)[0] >= 8:
            torch.set_float32_matmul_precision('high') # with bitnet this is not needed but it is good to have for future use ok xD I like yoir thinking bitnet is not the future it is the present and the future i love it xD ok 
        # Clear cache to start fresh
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # I love this line it is so good for memory management i wouldnt think to do it but it is so good
    
    global encodec_model, vocab_size, MASK_TOKEN_ID
    encodec_model = EncodecModel.encodec_model_24khz().to(DEVICE)
    encodec_model.set_target_bandwidth(ENCODEC_BANDWIDTH)  # Use higher quality encoding
    encodec_model.eval()
    
    # Note: Higher bandwidth = larger vocab size
    vocab_size = encodec_model.quantizer.bins + 1
    MASK_TOKEN_ID = encodec_model.quantizer.bins

    if RANK == 0:
        print(f"EnCodec model loaded with {ENCODEC_BANDWIDTH} kbps bandwidth")
        print(f"Vocab size: {vocab_size-1}, MASK_TOKEN_ID: {MASK_TOKEN_ID}")
        print(f"Quality improvement: {ENCODEC_BANDWIDTH/6.0:.1f}x higher bitrate than default")
        print(f"Audio enhancement: {'Enabled' if USE_AUDIO_ENHANCEMENT else 'Disabled'}")
        if USE_AUDIO_ENHANCEMENT:
            print("  - High-quality resampling")
            print("  - Audio normalization")
            print("  - DC offset removal")
            print("  - Soft compression post-processing")

    # --- NEW: Use VCTK Dataset ---
    if not DATASET_PATH.exists():
        if RANK == 0:
            print(f"Error: VCTK dataset not found at {DATASET_PATH}")
            print("Please download it from https://datashare.ed.ac.uk/handle/10283/3443")
            print("Alternatively, modify DATASET_PATH to point to your VCTK dataset location")
        cleanup_ddp()
        return

    try:
        full_dataset = VCTKPairDataset(root_path=DATASET_PATH, block_size=BLOCK_SIZE)
        if len(full_dataset) == 0:
            if RANK == 0:
                print(f"Error: No valid audio files found in {DATASET_PATH}")
                print("Please check that the dataset structure is correct")
            cleanup_ddp()
            return
        
        # Additional validation for dataset
        if len(full_dataset.audio_files) == 0:
            if RANK == 0:
                print(f"Error: No audio files found in {DATASET_PATH}")
                print("Please check that the dataset path contains audio files")
            cleanup_ddp()
            return
            
        if RANK == 0:
            print(f"Dataset loaded successfully!")
            print(f"Total audio files: {len(full_dataset.audio_files)}")
            print(f"Total training samples: {len(full_dataset)}")
            
            # Show sample files for verification
            if full_dataset.audio_files:
                print(f"Sample files: {[f.name for f in full_dataset.audio_files[:3]]}")
            
    except Exception as e:
        if RANK == 0:
            print(f"Error loading VCTK dataset: {e}")
            print("Please check the dataset path and structure")
            print("Expected structure: VCTK-Corpus-0.92/wav48_silence_trimmed/{speaker_id}/")
        cleanup_ddp()
        return

    # Split manually for simplicity
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if WORLD_SIZE > 1 else None
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # this may need to be adjusted based on your system and speed requirements

    writer = SummaryWriter(log_dir=LOG_DIR / f"run_{time.strftime('%Y%m%d-%H%M%S')}") if RANK == 0 else None

    # Initialize GradScaler for mixed precision if enabled
    scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION and torch.cuda.is_available() else None

    # --- NEW: Instantiate Encoder, Autoregressive Decoder, and Conditional Diffusion Decoder ---
    encoder = SpeechEncoder(vocab_size, N_EMBED, n_l=4, n_h=N_HEAD, block_sz=BLOCK_SIZE, dropout_val=DROPOUT).to(DEVICE)
    
    if USE_HYBRID_TRAINING:
        ar_decoder = AutoregressiveDecoder(vocab_size, N_EMBED, N_LAYER//2, N_HEAD, BLOCK_SIZE, DROPOUT).to(DEVICE)
        diffusion_decoder = ConditionalAudioConformerUNet(vocab_size, N_EMBED, N_LAYER, N_HEAD, BLOCK_SIZE, DROPOUT).to(DEVICE)
        
        if RANK == 0:
            encoder_params = sum(p.numel() for p in encoder.parameters()) / 1e6
            ar_params = sum(p.numel() for p in ar_decoder.parameters()) / 1e6
            diff_params = sum(p.numel() for p in diffusion_decoder.parameters()) / 1e6
            total_params = encoder_params + ar_params + diff_params
            print(f"=== Hybrid S2S Model Architecture ===")
            print(f"Encoder Parameters: {encoder_params:.2f}M")
            print(f"Autoregressive Decoder Parameters: {ar_params:.2f}M")
            print(f"Diffusion Decoder Parameters: {diff_params:.2f}M")
            print(f"Total Parameters: {total_params:.2f}M")
            print(f"Sequence Length: {BLOCK_SIZE} tokens (~{BLOCK_SIZE*3/100:.1f} seconds)")
    else:
        # Fallback to diffusion-only
        diffusion_decoder = ConditionalAudioConformerUNet(vocab_size, N_EMBED, N_LAYER, N_HEAD, BLOCK_SIZE, DROPOUT).to(DEVICE)
        ar_decoder = None
        
        if RANK == 0:
            encoder_params = sum(p.numel() for p in encoder.parameters()) / 1e6
            decoder_params = sum(p.numel() for p in diffusion_decoder.parameters()) / 1e6
            print(f"Encoder Parameters: {encoder_params:.2f}M")
            print(f"Decoder Parameters: {decoder_params:.2f}M")
            print(f"Total Parameters: {encoder_params + decoder_params:.2f}M")
    if USE_TORCH_COMPILE:
        try:
            if RANK == 0: print("Attempting to compile models with torch.compile...")
            encoder = torch.compile(encoder, mode=COMPILE_MODE, fullgraph=False, dynamic=False)
            if USE_HYBRID_TRAINING and ar_decoder is not None:
                ar_decoder = torch.compile(ar_decoder, mode=COMPILE_MODE, fullgraph=False, dynamic=False)
            diffusion_decoder = torch.compile(diffusion_decoder, mode=COMPILE_MODE, fullgraph=False, dynamic=False)
            if RANK == 0: print("Models compiled successfully.")
        except Exception as e:
            if RANK == 0: 
                print(f"torch.compile failed: {e}. Proceeding without compilation.")
                print("This is often due to GPU compatibility issues with advanced optimizations.")
                print("Common causes: MoE layers, gradient checkpointing, complex skip connections")
    else:
        if RANK == 0:
            print("torch.compile disabled - running in eager mode for stability.")

    if WORLD_SIZE > 1:
        encoder = DDP(encoder, device_ids=[LOCAL_RANK])
        if USE_HYBRID_TRAINING and ar_decoder is not None:
            ar_decoder = DDP(ar_decoder, device_ids=[LOCAL_RANK], find_unused_parameters=True)
        diffusion_decoder = DDP(diffusion_decoder, device_ids=[LOCAL_RANK], find_unused_parameters=True)

    # Combine parameters for the optimizer
    all_params = list(encoder.parameters()) + list(diffusion_decoder.parameters())
    if USE_HYBRID_TRAINING and ar_decoder is not None:
        all_params.extend(list(ar_decoder.parameters()))
    optimizer = torch.optim.AdamW(all_params, lr=LEARNING_RATE)
    
    # Improved scheduler with warmup and min learning rate
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_ITERS, eta_min=MIN_LEARNING_RATE)

    train_losses, val_losses, learning_rates, eval_iters_plot = [], [], [], []
    
    # Initialize loss tracking
    current_losses = {
        'total': [],
        'autoregressive': [],
        'diffusion': []
    }

    print("Starting S2S training...")
    if USE_HYBRID_TRAINING:
        if RANK == 0:
            print("=== HYBRID TRAINING MODE ===")
            print(f"Autoregressive Weight: {AUTOREGRESSIVE_WEIGHT}")
            print(f"Diffusion Weight: {DIFFUSION_WEIGHT}")
            print(f"Chunk Size: {AR_CHUNK_SIZE}, Overlap: {SEQUENCE_OVERLAP}")
            print(f"Sequence Length: {BLOCK_SIZE} tokens (~{BLOCK_SIZE/50:.1f} seconds)")
    else:
        if RANK == 0:
            print("=== DIFFUSION-ONLY MODE ===")
    for iter_num in range(MAX_ITERS):
        # Clear CUDA cache periodically for stability
        if iter_num % 100 == 0 and torch.cuda.is_available() and CLEAR_CACHE:
            torch.cuda.empty_cache()
            
        encoder.train()
        if USE_HYBRID_TRAINING and ar_decoder is not None:
            ar_decoder.train()
        diffusion_decoder.train()

        # --- MODIFIED: Hybrid Training Step ---
        try:
            # Mark CUDA graph step if using torch.compile (safety measure)
            if USE_TORCH_COMPILE and torch.cuda.is_available():
                torch.compiler.cudagraph_mark_step_begin()
                
            batch = next(iter(train_dataloader))
            source_tokens, target_tokens = batch
            source_tokens, target_tokens = source_tokens.to(DEVICE), target_tokens.to(DEVICE)
        except Exception as e:
            if RANK == 0:
                print(f"Error loading batch: {e}")
            continue
        
        # Get conditioning vector from the source audio
        conditioning_vector = encoder(source_tokens)

        # Use mixed precision if enabled
        if USE_MIXED_PRECISION and scaler is not None:
            # Extract device type as string from device object
            device_type = DEVICE.type if hasattr(DEVICE, 'type') else 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.amp.autocast(device_type=device_type):
                if USE_HYBRID_TRAINING and ar_decoder is not None:
                    # Hybrid training: Autoregressive + Diffusion losses
                    ar_loss = compute_chunked_autoregressive_loss(ar_decoder, source_tokens, target_tokens, conditioning_vector)
                    diffusion_loss = compute_diffusion_loss(diffusion_decoder, source_tokens, target_tokens, conditioning_vector)
                    
                    # Combine losses
                    total_loss = AUTOREGRESSIVE_WEIGHT * ar_loss + DIFFUSION_WEIGHT * diffusion_loss
                    loss = total_loss  # For logging compatibility
                else:
                    # Fallback to diffusion-only training
                    diffusion_loss = compute_diffusion_loss(diffusion_decoder, source_tokens, target_tokens, conditioning_vector)
                    loss = diffusion_loss
                    ar_loss = None
        else:
            if USE_HYBRID_TRAINING and ar_decoder is not None:
                # Hybrid training: Autoregressive + Diffusion losses
                ar_loss = compute_chunked_autoregressive_loss(ar_decoder, source_tokens, target_tokens, conditioning_vector)
                diffusion_loss = compute_diffusion_loss(diffusion_decoder, source_tokens, target_tokens, conditioning_vector)
                
                # Combine losses
                total_loss = AUTOREGRESSIVE_WEIGHT * ar_loss + DIFFUSION_WEIGHT * diffusion_loss
                loss = total_loss  # For logging compatibility
            else:
                # Fallback to diffusion-only training
                diffusion_loss = compute_diffusion_loss(diffusion_decoder, source_tokens, target_tokens, conditioning_vector)
                loss = diffusion_loss
                ar_loss = None
        
        # Track losses for better monitoring
        current_losses['total'].append(loss.item())
        if USE_HYBRID_TRAINING and ar_loss is not None:
            current_losses['autoregressive'].append(ar_loss.item())
            current_losses['diffusion'].append(diffusion_loss.item())
        
        # Keep only recent losses for averaging
        for key in current_losses:
            if len(current_losses[key]) > LOSS_HISTORY_SIZE:
                current_losses[key] = current_losses[key][-LOSS_HISTORY_SIZE:]
        
        optimizer.zero_grad(set_to_none=True)
        
        # Backward pass with mixed precision and gradient clipping
        if USE_MIXED_PRECISION and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, GRADIENT_CLIP_VAL)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, GRADIENT_CLIP_VAL)
            optimizer.step()
        
        # Learning rate warmup
        if iter_num < WARMUP_STEPS:
            lr_scale = min(1.0, iter_num / WARMUP_STEPS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE * lr_scale
        else:
            scheduler.step()

        # --- MODIFIED: Evaluation Step ---
        if iter_num % EVAL_INTERVAL == 0:
            encoder.eval()
            if USE_HYBRID_TRAINING and ar_decoder is not None:
                ar_decoder.eval()
            diffusion_decoder.eval()
            avg_val_loss = 0.
            val_count = 0
            with torch.no_grad():
                try:
                    for val_batch in val_dataloader:
                        # Mark CUDA graph step if using torch.compile (safety measure)
                        if USE_TORCH_COMPILE and torch.cuda.is_available():
                            torch.compiler.cudagraph_mark_step_begin()
                            
                        val_source, val_target = val_batch
                        val_source, val_target = val_source.to(DEVICE), val_target.to(DEVICE)
                        
                        val_conditioning_vector = encoder(val_source)
                        
                        if USE_HYBRID_TRAINING and ar_decoder is not None:
                            # Compute both losses for validation
                            val_ar_loss = compute_chunked_autoregressive_loss(ar_decoder, val_source, val_target, val_conditioning_vector)
                            val_diff_loss = compute_diffusion_loss(diffusion_decoder, val_source, val_target, val_conditioning_vector)
                            val_loss_batch = AUTOREGRESSIVE_WEIGHT * val_ar_loss + DIFFUSION_WEIGHT * val_diff_loss
                        else:
                            # Diffusion-only validation
                            val_loss_batch = compute_diffusion_loss(diffusion_decoder, val_source, val_target, val_conditioning_vector)
                        
                        avg_val_loss += val_loss_batch.item()
                        val_count += 1
                        if val_count >= 10:  # Limit validation batches for speed
                            break
                except Exception as e:
                    if RANK == 0:
                        print(f"Error during validation: {e}")
                        avg_val_loss = float(9e10)  # Set high loss to indicate failure not infinity as that causes the training to stop
                        val_count = 1
            
            if val_count > 0:
                avg_val_loss /= val_count

            if RANK == 0:
                current_lr = optimizer.param_groups[0]['lr']
                
                # Calculate moving averages for better insight
                avg_total_loss = sum(current_losses['total']) / len(current_losses['total']) if current_losses['total'] else loss.item()
                
                if USE_HYBRID_TRAINING and ar_decoder is not None:
                    avg_ar_loss = sum(current_losses['autoregressive']) / len(current_losses['autoregressive']) if current_losses['autoregressive'] else 0.0
                    avg_diff_loss = sum(current_losses['diffusion']) / len(current_losses['diffusion']) if current_losses['diffusion'] else 0.0
                    
                    print(f"Iter {iter_num}/{MAX_ITERS} | Total: {avg_total_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.2e}")
                    print(f"  AR Loss: {avg_ar_loss:.4f} | Diff Loss: {avg_diff_loss:.4f} | AR/Diff Ratio: {avg_ar_loss/avg_diff_loss:.2f}")
                else:
                    print(f"Iter {iter_num}/{MAX_ITERS} | Train Loss: {avg_total_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")
                
                # --- Generate a sample for validation ---
                if iter_num > 0:  # Skip sampling at iteration 0
                    try:
                        # Get a sample batch from val loader
                        val_source, val_target = next(iter(val_dataloader))
                        val_source, val_target = val_source.to(DEVICE), val_target.to(DEVICE)

                        with torch.no_grad():
                            encoder_to_gen = encoder.module if isinstance(encoder, DDP) else encoder
                            diffusion_decoder_to_gen = diffusion_decoder.module if isinstance(diffusion_decoder, DDP) else diffusion_decoder

                            # Get condition from the first item in the validation batch
                            conditioning_vector_val = encoder_to_gen(val_source[0:1])

                            # Generate using enhanced DDIM with clean sampling (use diffusion decoder for high quality)
                            if CLEAN_SAMPLING_MODE:
                                # Multiple sampling passes for best quality
                                best_sample = None
                                best_score = float('-inf')
                                
                                for sample_attempt in range(3):  # Try 3 times, pick best
                                    temp = SAMPLING_TEMPERATURE * (0.8 + 0.4 * sample_attempt / 2)  # Vary temperature
                                    sample = ddim_sampler(diffusion_decoder_to_gen, 
                                                        shape=(1, BLOCK_SIZE), 
                                                        conditioning_vector=conditioning_vector_val,
                                                        temperature=temp)
                                    
                                    # Score based on token diversity and validity
                                    unique_tokens = len(torch.unique(sample))
                                    valid_range = (sample >= 0).all() and (sample < vocab_size - 1).all()
                                    score = unique_tokens if valid_range else -1
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_sample = sample
                                
                                generated_codes_tensor = best_sample
                            else:
                                # Standard single-pass sampling
                                generated_codes_tensor = ddim_sampler(diffusion_decoder_to_gen, 
                                                                   shape=(1, BLOCK_SIZE), 
                                                                   conditioning_vector=conditioning_vector_val)
                        
                            # Validate generated codes and fix any out-of-range values
                            max_code = generated_codes_tensor.max().item()
                            min_code = generated_codes_tensor.min().item()
                            
                            if max_code >= vocab_size - 1 or min_code < 0:
                                print(f"Warning: Generated codes range [{min_code}, {max_code}] exceeds vocab range [0, {vocab_size-2}]. Clipping...")
                                generated_codes_tensor = torch.clamp(generated_codes_tensor, 0, vocab_size - 2)
                                
                            # Double-check after clipping
                            max_code_after = generated_codes_tensor.max().item()
                            min_code_after = generated_codes_tensor.min().item()
                            print(f"Codes after clipping: range [{min_code_after}, {max_code_after}]")

                            # --- Decode source, target, and generated audio for comparison ---
                            try:
                                # Source audio
                                source_codes = torch.zeros(1, 1, BLOCK_SIZE, dtype=torch.long, device=DEVICE)
                                source_codes[0, 0, :] = torch.clamp(val_source[0], 0, vocab_size - 2)  # Clamp source too
                                source_frames = [(source_codes, None)]
                                with torch.no_grad():
                                    source_audio = encodec_model.decode(source_frames)
                                
                                # Target audio  
                                target_codes = torch.zeros(1, 1, BLOCK_SIZE, dtype=torch.long, device=DEVICE)
                                target_codes[0, 0, :] = torch.clamp(val_target[0], 0, vocab_size - 2)  # Clamp target too
                                target_frames = [(target_codes, None)]
                                with torch.no_grad():
                                    target_audio = encodec_model.decode(target_frames)
                                
                                # Generated audio with post-processing
                                generated_codes = torch.zeros(1, 1, BLOCK_SIZE, dtype=torch.long, device=DEVICE)
                                
                                # Apply token smoothing for cleaner audio
                                if CLEAN_SAMPLING_MODE:
                                    # Remove outlier tokens (simple median filter)
                                    codes_to_process = generated_codes_tensor[0].clone()
                                    for i in range(2, len(codes_to_process) - 2):
                                        window = codes_to_process[i-2:i+3]
                                        median_val = torch.median(window)
                                        # Replace outliers with median
                                        if torch.abs(codes_to_process[i] - median_val) > vocab_size * 0.1:
                                            codes_to_process[i] = median_val
                                    generated_codes[0, 0, :] = codes_to_process
                                else:
                                    generated_codes[0, 0, :] = generated_codes_tensor[0].to(DEVICE)
                                
                                generated_frames = [(generated_codes, None)]
                                with torch.no_grad():
                                    generated_audio = encodec_model.decode(generated_frames)
                                    
                                    # Enhanced audio post-processing for better quality
                                    if USE_AUDIO_ENHANCEMENT and CLEAN_SAMPLING_MODE:
                                        # Apply gentle processing to improve perceived quality
                                        audio_tensor = generated_audio.squeeze()
                                        
                                        # 1. Normalize to prevent clipping
                                        if AUDIO_NORMALIZATION:
                                            max_val = torch.max(torch.abs(audio_tensor))
                                            if max_val > 0.95:
                                                audio_tensor = audio_tensor * (0.95 / max_val)
                                        
                                        # 2. Simple DC offset removal
                                        audio_tensor = audio_tensor - torch.mean(audio_tensor)
                                        
                                        # 3. Gentle compression for more consistent levels
                                        # Soft-knee compressor simulation
                                        threshold = 0.7
                                        ratio = 0.3
                                        abs_audio = torch.abs(audio_tensor)
                                        compress_mask = abs_audio > threshold
                                        compressed_vals = threshold + (abs_audio[compress_mask] - threshold) * ratio
                                        audio_tensor[compress_mask] = torch.sign(audio_tensor[compress_mask]) * compressed_vals
                                        
                                        generated_audio = audio_tensor.unsqueeze(0).unsqueeze(0)
                                    
                                    # Ensure audio is in reasonable range
                                    generated_audio = torch.clamp(generated_audio, -1.0, 1.0)

                                # Save all three audio files for comparison
                                source_path = AUDIO_SAMPLES_DIR / f"source_iter_{iter_num}.wav"
                                target_path = AUDIO_SAMPLES_DIR / f"target_iter_{iter_num}.wav"
                                generated_path = AUDIO_SAMPLES_DIR / f"generated_iter_{iter_num}.wav"
                                
                                torchaudio.save(str(source_path), source_audio.squeeze(0).cpu(), encodec_model.sample_rate)
                                torchaudio.save(str(target_path), target_audio.squeeze(0).cpu(), encodec_model.sample_rate)
                                torchaudio.save(str(generated_path), generated_audio.squeeze(0).cpu(), encodec_model.sample_rate)
                                
                                print(f"Iter {iter_num}: Generated S2S validation samples:")
                                print(f"  Source: {source_path}")
                                print(f"  Target: {target_path}")
                                print(f"  Generated: {generated_path}")
                                print(f"  Note: Source and target should be same utterance from different speakers")
                                
                                if writer:
                                    writer.add_audio('Source_Audio', source_audio.squeeze(0).cpu(), iter_num, sample_rate=encodec_model.sample_rate)
                                    writer.add_audio('Target_Audio', target_audio.squeeze(0).cpu(), iter_num, sample_rate=encodec_model.sample_rate)
                                    writer.add_audio('Generated_Audio', generated_audio.squeeze(0).cpu(), iter_num, sample_rate=encodec_model.sample_rate)
                                    
                            except Exception as decode_error:
                                print(f"Error during EnCodec decoding: {decode_error}")
                                print("Skipping audio generation for this iteration...")
                    except Exception as e:
                        print(f"Error generating S2S audio sample: {e}")
                        traceback.print_exc()
                
                # Always log training metrics
                if writer:
                     writer.add_scalar('Loss/train', loss.item(), iter_num)
                     writer.add_scalar('Loss/val', avg_val_loss, iter_num)

        if RANK == 0 and iter_num > 0 and iter_num % CHECKPOINT_INTERVAL == 0:
            # Checkpointing logic for encoder and decoders
            ckpt_path = CHECKPOINT_DIR / f"s2s_hybrid_ckpt_iter_{iter_num}.pt"
            
            encoder_to_save = encoder.module if isinstance(encoder, DDP) else encoder
            diffusion_decoder_to_save = diffusion_decoder.module if isinstance(diffusion_decoder, DDP) else diffusion_decoder
            
            encoder_state = encoder_to_save._orig_mod.state_dict() if hasattr(encoder_to_save, '_orig_mod') else encoder_to_save.state_dict()
            diffusion_decoder_state = diffusion_decoder_to_save._orig_mod.state_dict() if hasattr(diffusion_decoder_to_save, '_orig_mod') else diffusion_decoder_to_save.state_dict()
            
            checkpoint_data = {
                'iteration': iter_num,
                'encoder_state_dict': encoder_state,
                'diffusion_decoder_state_dict': diffusion_decoder_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'use_hybrid_training': USE_HYBRID_TRAINING
            }
            
            # Save mixed precision scaler state if available
            if USE_MIXED_PRECISION and scaler is not None:
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()
            
            if USE_HYBRID_TRAINING and ar_decoder is not None:
                ar_decoder_to_save = ar_decoder.module if isinstance(ar_decoder, DDP) else ar_decoder
                ar_decoder_state = ar_decoder_to_save._orig_mod.state_dict() if hasattr(ar_decoder_to_save, '_orig_mod') else ar_decoder_to_save.state_dict()
                checkpoint_data['ar_decoder_state_dict'] = ar_decoder_state
            
            torch.save(checkpoint_data, ckpt_path)
            print(f"Saved S2S {'hybrid' if USE_HYBRID_TRAINING else 'diffusion'} checkpoint to {ckpt_path}")
    
    if RANK == 0 and writer:
        writer.close()
    if WORLD_SIZE > 1:
        cleanup_ddp()

# Helper for get_batch_from_loader
def get_batch_from_loader(loader_iter, loader, epoch_for_sampler):
    try:
        batch = next(loader_iter)
    except StopIteration:
        if RANK == 0: 
            print(f"Resetting dataloader iterator (epoch {epoch_for_sampler}).")
        if WORLD_SIZE > 1 and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch_for_sampler)
        loader_iter = iter(loader)
        batch = next(loader_iter)
    except Exception as e:
        if RANK == 0:
            print(f"Error in get_batch_from_loader: {e}")
        raise e
    return tuple(t.to(DEVICE) for t in batch), loader_iter

if __name__ == "__main__":
    main()