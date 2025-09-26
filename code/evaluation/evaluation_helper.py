# evaluation_helper.py

import sys
from pathlib import Path
import torch
import whisper
import numpy as np
import pandas as pd
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_paths():
    """Setup all necessary paths and environment variables"""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    clarity_path = project_root / "code" / "audio_files" / "16981327" / "clarity"
    baseline_path = clarity_path / "recipes" / "cad_icassp_2026" / "baseline"
    if not clarity_path.exists():
        raise FileNotFoundError(f"Clarity path not found: {clarity_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline path not found: {baseline_path}")
    if str(clarity_path) not in sys.path:
        sys.path.append(str(clarity_path))
    if str(baseline_path) not in sys.path:
        sys.path.append(str(baseline_path))
    logging.info(f"Added to Python path: {clarity_path}")
    logging.info(f"Added to Python path: {baseline_path}")
    return project_root, clarity_path, baseline_path

project_root, clarity_path, baseline_path = setup_paths()
PATH_TO_BASELINE_CODE = baseline_path

from transcription_scorer import SentenceScorer
from shared_predict_utils import LogisticModel
from evaluate import compute_scores

class EvaluationHelper:
    """A helper class to wrap the baseline evaluation logic."""
    def __init__(self, whisper_version="base.en", load_model=False):
        # We can optionally skip loading the model if we're not using it
        if load_model:
            print("Initializing Evaluation Helper...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            self.asr_model = whisper.load_model(whisper_version, device=device)
            contractions_file = PATH_TO_BASELINE_CODE / "contractions.csv"
            self.scorer = SentenceScorer(str(contractions_file))
            print("Initialization complete.")
        else:
            print("Evaluation Helper initialized without loading Whisper model.")

    # The score_single_file method is no longer needed for this test run,
    # but we'll keep it for when you evaluate your own audio later.
    def score_single_file(self, audio_path: Path, reference_lyrics: str) -> float:
        # ... (this function remains the same)
        pass

    def run_full_evaluation_from_precomputed(self, 
                            precomputed_valid_scores_path: Path, # <-- New parameter
                            metadata_path: Path, 
                            train_metadata_path: Path,
                            precomputed_train_scores_path: Path):
        """
        Runs the evaluation pipeline from Stage 2 onwards, using precomputed scores.
        """
        print("--- Starting Evaluation from Precomputed Scores ---")
        
        # --- STAGE 1: Load Precomputed Scores (SKIPPING AUDIO PROCESSING) ---
        print("\n[Stage 1/3] SKIPPING Whisper processing. Loading precomputed validation scores...")
        
        # Manually read the JSON Lines file for robustness
        data_list = []
        with open(precomputed_valid_scores_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line))
        valid_scores_df = pd.DataFrame(data_list)
        print(f"Loaded {len(valid_scores_df)} precomputed validation scores.")
        
        # --- The slow Stage 1 loop is now gone ---

        # --- STAGE 2: Calibrate Predictions ---
        print("\n[Stage 2/3] Calibrating predictions using logistic model...")
        
        # This part remains exactly the same
        data_list_train = []
        with open(precomputed_train_scores_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list_train.append(json.loads(line))
        train_scores_df = pd.DataFrame(data_list_train)

        metadata_df = pd.read_json(metadata_path)
        train_metadata_df = pd.read_json(train_metadata_path)
        
        train_df = pd.merge(train_scores_df, train_metadata_df, on='signal')
        
        model = LogisticModel()
        model.fit(train_df["whisper"], train_df.correctness)
        print("Logistic model fitted on training data.")
        
        # The column name from the precomputed file is 'whisper.mixture'
        # We rename it to match what the rest of the script expects.
        if 'whisper.mixture' in valid_scores_df.columns:
            valid_scores_df = valid_scores_df.rename(columns={'whisper.mixture': 'whisper'})
        
        valid_scores_df["predicted"] = model.predict(valid_scores_df.whisper)
        
        # --- STAGE 3: Evaluate Final Scores ---
        print("\n[Stage 3/3] Computing final RMSE and NCC scores...")
        
        final_df = pd.merge(valid_scores_df, metadata_df, on='signal')
        
        # This can happen if the metadata has signals the scores file doesn't
        if len(final_df) == 0:
            print("CRITICAL ERROR: No matching signals found between scores and metadata.")
            return None
            
        scores = compute_scores(final_df["predicted"], final_df["correctness"])
        
        print("\n--- Evaluation Complete! ---")
        print(json.dumps(scores, indent=2))
        return scores


if __name__ == '__main__':
    project_root = setup_paths()[0]
    
    cadenza_data_root = project_root / "code" / "audio_files" / "16981327" / "cadenza_data"
    
    # --- Paths for the metadata files ---
    valid_metadata = cadenza_data_root / "metadata" / "valid_metadata.json"
    train_metadata = cadenza_data_root / "metadata" / "train_metadata.json"

    # --- Paths for the PRECOMPUTED scores ---
    precomputed_train_scores = PATH_TO_BASELINE_CODE / "precomputed" / "cadenza_data.train.whisper.mixture.jsonl"
    # This is the new file we are using to skip the slow part
    precomputed_valid_scores = PATH_TO_BASELINE_CODE / "precomputed" / "cadenza_data.valid.whisper.mixture.jsonl"

    # Create the helper, skip loading the heavy Whisper model to save time
    eval_helper = EvaluationHelper(load_model=False) 
    
    # Run the evaluation process using only precomputed files
    final_scores = eval_helper.run_full_evaluation_from_precomputed(
        precomputed_valid_scores_path=precomputed_valid_scores,
        metadata_path=valid_metadata,
        train_metadata_path=train_metadata,
        precomputed_train_scores_path=precomputed_train_scores
    )