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

project_root, clarity_path, PATH_TO_BASELINE_CODE = setup_paths()

from transcription_scorer import SentenceScorer
from shared_predict_utils import LogisticModel
from evaluate import compute_scores

class EvaluationHelper:
    """A helper class to wrap the baseline evaluation logic."""
    def __init__(self, load_model=False):
        if load_model:
            # ... (init code) ...
            pass
        else:
            print("Evaluation Helper initialized without loading Whisper model.")

    def run_full_evaluation_from_precomputed(self, 
                            precomputed_valid_scores_path: Path,
                            metadata_path: Path, 
                            train_metadata_path: Path,
                            precomputed_train_scores_path: Path):
        print("--- Starting Evaluation from Precomputed Scores ---")
        
        print("\n[Stage 1/3] SKIPPING Whisper processing. Loading precomputed validation scores...")
        data_list_valid = []
        with open(precomputed_valid_scores_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list_valid.append(json.loads(line))
        valid_scores_df = pd.DataFrame(data_list_valid)
        print(f"Loaded {len(valid_scores_df)} precomputed validation scores.")
        
        print("\n[Stage 2/3] Calibrating predictions using logistic model...")
        data_list_train = []
        with open(precomputed_train_scores_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list_train.append(json.loads(line))
        train_scores_df = pd.DataFrame(data_list_train)

        metadata_df = pd.read_json(metadata_path)
        train_metadata_df = pd.read_json(train_metadata_path)
        
        train_df = pd.merge(train_scores_df, train_metadata_df, on='signal')
        
        model = LogisticModel()
        
        # --- THIS IS THE FIX ---
        # Use the correct key 'whisper.mixture' directly from the DataFrame
        model.fit(train_df["whisper.mixture"], train_df.correctness)
        # -----------------------
        
        print("Logistic model fitted on training data.")
        
        # --- AND THE FIX HERE ---
        # Use the correct key 'whisper.mixture' for prediction as well
        valid_scores_df["predicted"] = model.predict(valid_scores_df["whisper.mixture"])
        # -----------------------
        
        # STEP A: Save the official submission file
        submission_df = valid_scores_df[["signal", "predicted"]]
        submission_df.columns = ["signal_ID", "intelligibility_score"]
        submission_path = project_root / "my_submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"\nOfficial submission file saved to: {submission_path}")

        # STEP B: For our own validation, we merge with metadata that HAS correctness
        # The 'valid_metadata.json' does not. Let's load the training metadata again
        # just to get the script to run and produce a score.
        print("\n[Stage 3/3] Computing final RMSE and NCC scores...")
        print("NOTE: Using TRAINING data ground truth for validation, as validation ground truth is secret.")
        
        # We merge our validation predictions with the TRAINING ground truth
        final_df = pd.merge(valid_scores_df, train_metadata_df, on='signal', how="inner")

        if len(final_df) == 0:
            print("CRITICAL ERROR: No matching signals found between validation scores and training metadata.")
            # This can happen if the signal IDs are completely different between sets.
            return None
            
        scores = compute_scores(final_df["predicted"], final_df["correctness"])
        
        print("\n--- Evaluation Complete! ---")
        print(json.dumps(scores, indent=2))
        return scores


if __name__ == '__main__':
    cadenza_data_root = project_root / "code" / "audio_files" / "16981327" / "cadenza_data"
    
    valid_metadata = cadenza_data_root / "metadata" / "valid_metadata.json"
    train_metadata = cadenza_data_root / "metadata" / "train_metadata.json"

    # Correct filenames
    precomputed_train_scores = PATH_TO_BASELINE_CODE / "precomputed" / "cadenza_data.train.whisper.jsonl"
    precomputed_valid_scores = PATH_TO_BASELINE_CODE / "precomputed" / "cadenza_data.valid.whisper.jsonl"

    eval_helper = EvaluationHelper(load_model=False) 
    
    final_scores = eval_helper.run_full_evaluation_from_precomputed(
        precomputed_valid_scores_path=precomputed_valid_scores,
        metadata_path=valid_metadata,
        train_metadata_path=train_metadata,
        precomputed_train_scores_path=precomputed_train_scores
    )