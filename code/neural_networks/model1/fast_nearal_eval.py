import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import json
import logging
from tqdm import tqdm
import soundfile as sf
import torchaudio
import random

#TODO the import FOR EVAL HELPER IS BROCKEN  WARNING IT WILL JUST GIVE AUDIO FILES NOT EVAL

# --- This script is self-contained. All necessary code is included here. ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. SETUP: ROBUST PATHING AND IMPORTS ---

def setup_paths():
    """
    Finds the project root by searching upwards from the current file.
    This is the most robust way to handle pathing.
    """
    current_file = Path(__file__).resolve()
    project_root = current_file
    # We will search for a reliable anchor, the 'code' directory.
    while not (project_root / 'code').exists() and project_root.parent != project_root:
        project_root = project_root.parent
        
    if not (project_root / 'code').exists():
        raise FileNotFoundError("Could not find the project root. Make sure this script is inside your project.")

    code_path = project_root / "code"
    baseline_path = code_path / "audio_files" / "16981327" / "clarity" / "recipes" / "cad_icassp_2026" / "baseline"
    
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))
    if str(baseline_path) not in sys.path:
        sys.path.insert(0, str(baseline_path))
        
    logging.info(f"Project root is: {project_root}")
    return project_root, baseline_path

project_root, baseline_code_path = setup_paths()

# Now, with the correct path set, these imports will work
from data_loading import create_dataloaders
from neural_networks.model_gen.model import SpeechEncoder, ConditionalAudioConformerUNet
from encodec import EncodecModel
from encodec.utils import convert_audio

# --- 2. THE S2S MODEL INFERENCE CLASS (SELF-CONTAINED) ---

class S2S_Model:
    """A self-contained class for loading and running your trained S2S model."""
    def __init__(self, checkpoint_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Initializing S2S Model on device: {self.device}")

        self.encodec_model = EncodecModel.encodec_model_24khz().to(self.device)
        self.encodec_model.set_target_bandwidth(12.0)
        self.vocab_size = self.encodec_model.quantizer.bins + 1
        self.mask_token_id = self.encodec_model.quantizer.bins
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        n_embed, n_head, n_layer, block_size, dropout = 64, 4, 8, 256, 0.05
        self.encoder = SpeechEncoder(self.vocab_size, n_embed, 4, n_head, block_size, dropout).to(self.device)
        self.diffusion_decoder = ConditionalAudioConformerUNet(self.vocab_size, n_embed, n_layer, n_head, block_size, dropout).to(self.device)
        self.hearing_proj = torch.nn.Linear(1, n_embed).to(self.device)

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.diffusion_decoder.load_state_dict(checkpoint['diffusion_decoder_state_dict'])
        if 'hearing_proj' in checkpoint and checkpoint['hearing_proj'] is not None:
             self.hearing_proj.load_state_dict(checkpoint['hearing_proj'])

        self.encoder.eval()
        self.diffusion_decoder.eval()
        self.hearing_proj.eval()
        logging.info("S2S Model and weights loaded successfully.")

    def process_audio(self, signal_tensor: torch.Tensor, metadata: dict) -> torch.Tensor:
        """The main processing function."""
        with torch.no_grad():
            audio_for_enc = signal_tensor.to(self.device)
            if audio_for_enc.dim() == 1: audio_for_enc = audio_for_enc.unsqueeze(0)
            if audio_for_enc.shape[0] > 1: audio_for_enc = torch.mean(audio_for_enc, dim=0, keepdim=True)
            
            wav_conv = convert_audio(audio_for_enc, 44100, 24000, 1)
            encoded_frames = self.encodec_model.encode(wav_conv.unsqueeze(0))
            source_tokens = encoded_frames[0][0][0, :, :256]

            conditioning_vector = self.encoder(source_tokens)
            
            hearing_loss_label = metadata.get('hearing_loss', 'No Loss')
            mapping = {'No Loss': 0.0, 'Mild': 0.33, 'Moderate': 0.66, 'Severe': 1.0}
            hearing_val = torch.tensor([mapping.get(hearing_loss_label, 0.0)], device=self.device)
            hearing_embed = self.hearing_proj(hearing_val.unsqueeze(-1))
            conditioning_vector += hearing_embed
            
            shape = (1, 256)
            x_t = torch.full(shape, self.mask_token_id, dtype=torch.long, device=self.device)
            
            times = torch.linspace(199, 0, steps=50).long()
            for t in times:
                t_tensor = torch.full((1,), t, dtype=torch.long, device=self.device)
                pred_x0_logits = self.diffusion_decoder(x_t, t_tensor, conditioning_vector)
                pred_x0 = torch.argmax(pred_x0_logits, dim=-1)
                t_prev = t - (200 // 50)
                if t_prev < 0:
                    x_t = pred_x0; break
                
                corruption_probs = (t_prev.float() + 1) / 200
                should_corrupt = torch.rand_like(pred_x0, dtype=torch.float32) < corruption_probs
                mask = torch.full_like(pred_x0, self.mask_token_id)
                x_t = torch.where(should_corrupt, mask, pred_x0)
            
            generated_tokens = torch.clamp(x_t, 0, self.vocab_size - 2)
            generated_frames = [(generated_tokens.unsqueeze(0), None)]
            enhanced_audio = self.encodec_model.decode(generated_frames).squeeze(0).cpu()
            return torchaudio.functional.resample(enhanced_audio, 24000, 44100)

# --- 3. THE MAIN SCRIPT ---

def run_final_neural_experiment(num_samples: int = 400):
    """The all-in-one function to get your final score and be free."""
    
    logging.info(f"--- Starting Final Neural Evaluation on {num_samples} random training samples ---")
    
    # Define all paths
    cadenza_data_root = project_root / "code" / "audio_files" / "16981327" / "cadenza_data"
    checkpoint_path = project_root / "code" / "neural_networks" / "model1" / "model.pt"
    output_dir = project_root / "results" / f"FINAL_NEURAL_RUN_{num_samples}_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- PHASE 1: ENHANCEMENT ---
    logging.info("--- Phase 1: Enhancing Audio with Your Trained Model ---")
    s2s_model = S2S_Model(str(checkpoint_path))
    
    train_loader_full, _ = create_dataloaders(cadenza_data_root.parent, batch_size=4)
    random_indices = random.sample(range(len(train_loader_full.dataset)), num_samples)
    subset_dataset = Subset(train_loader_full.dataset, random_indices)
    fast_train_loader = DataLoader(subset_dataset, batch_size=4, collate_fn=train_loader_full.collate_fn)
    
    for batch in tqdm(fast_train_loader, desc="Enhancing Audio"):
        for i in range(len(batch['signal'])):
            try:
                metadata = batch['metadata'][i]
                signal_id = metadata['signal']
                processed_tensor = s2s_model.process_audio(batch['signal'][i], metadata)
                
                # --- THIS IS THE CRITICAL FILENAME FIX ---
                # Save the file with the name the evaluator expects: [signal_id].wav
                output_filename = output_dir / f"{signal_id}.wav"
                
                # Normalize and save
                proc_to_write = processed_tensor.numpy().T
                max_abs = np.max(np.abs(proc_to_write))
                if max_abs > 0: proc_to_write = proc_to_write / max_abs * 0.99
                sf.write(output_filename, proc_to_write, 44100)
            except Exception as e:
                logging.error(f"Failed to process signal {metadata.get('signal', 'unknown')}: {e}")

    logging.info(f"--- Phase 1 Complete. {num_samples} enhanced files saved to {output_dir} ---")
    
    # --- PHASE 2: EVALUATION ---
    logging.info("--- Phase 2: Evaluating Your Enhanced Audio ---")
    
    train_metadata = cadenza_data_root / "metadata" / "train_metadata.json"
    precomputed_train_scores = baseline_code_path / "precomputed" / "cadenza_data.train.whisper.jsonl"
    output_csv = project_root / f"submission_train_NEURAL_NETWORK_{num_samples}_samples.csv"
    
    eval_helper = EvaluationHelper()
    eval_helper.evaluate_audio_folder(
        audio_dir=output_dir,
        metadata_path=train_metadata,
        train_metadata_path=train_metadata,
        precomputed_train_scores_path=precomputed_train_scores,
        output_csv_path=output_csv
    )
    
    logging.info(f"--- Final CSV Created: {output_csv.name} ---")
    logging.info("Run 'evaluate_csv.py' to get your final score and be free.")


if __name__ == '__main__':
    run_final_neural_experiment(num_samples=400)