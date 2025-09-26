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
    # Get the absolute path to the project root
    current_file = Path(__file__).resolve()
    # This navigates UP from .../code/evaluation/ to the main project folder
    project_root = current_file.parent.parent.parent  
    
    # Path to the clarity baseline code
    clarity_path = project_root / "code" / "audio_files" / "16981327" / "clarity"
    baseline_path = clarity_path / "recipes" / "cad_icassp_2026" / "baseline"
    
    # Verify paths exist
    if not clarity_path.exists():
        raise FileNotFoundError(f"Clarity path not found: {clarity_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline path not found: {baseline_path}")
    
    # Add paths to system path
    if str(clarity_path) not in sys.path:
        sys.path.append(str(clarity_path))
    if str(baseline_path) not in sys.path:
        sys.path.append(str(baseline_path))
    
    logging.info(f"Added to Python path: {clarity_path}")
    logging.info(f"Added to Python path: {baseline_path}")
    
    return project_root, clarity_path, baseline_path

# Setup paths before imports
project_root, clarity_path, baseline_path = setup_paths()
PATH_TO_BASELINE_CODE = baseline_path


# Now we can import their tools directly
from transcription_scorer import SentenceScorer
from shared_predict_utils import LogisticModel
from evaluate import compute_scores

class EvaluationHelper:
    # ... (The entire class definition remains unchanged, it is already perfect) ...
    """A helper class to wrap the baseline evaluation logic."""

    def __init__(self, whisper_version="base.en"):
        print("Initializing Evaluation Helper...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # 1. Load the Whisper model once
        self.asr_model = whisper.load_model(whisper_version, device=device)
        
        # 2. Initialize the sentence scorer once
        contractions_file = PATH_TO_BASELINE_CODE / "contractions.csv"
        self.scorer = SentenceScorer(str(contractions_file))
        print("Initialization complete.")

    def score_single_file(self, audio_path: Path, reference_lyrics: str) -> float:
        """
        Computes the raw Whisper correctness score for a single audio file.
        This is a wrapper around the baseline's 'compute_correctness' logic.
        """
        # Whisper's transcribe function handles loading the audio
        hypothesis = self.asr_model.transcribe(
            str(audio_path), fp16=False, language="en", temperature=0.0
        )["text"]

        # Score the transcription using the baseline's scorer
        results = self.scorer.score([reference_lyrics], [hypothesis])
        total_words = results.substitutions + results.deletions + results.hits
        
        if total_words == 0:
            return 0.0
        
        return results.hits / total_words

    def run_full_evaluation(self, 
                            enhanced_audio_dir: Path, 
                            metadata_path: Path, 
                            train_metadata_path: Path,
                            precomputed_train_scores_path: Path):
        """
        Runs the full 3-stage evaluation pipeline on a folder of enhanced audio.
        """
        print(f"--- Starting Full Evaluation on folder: {enhanced_audio_dir} ---")
        
        # --- STAGE 1: Compute Raw Scores ---
        print("\n[Stage 1/3] Computing raw Whisper scores for your audio...")
        metadata_df = pd.read_json(metadata_path)
        raw_scores = []
        for _, row in metadata_df.iterrows():
            signal_id = row['signal']
            lyrics = row['prompt']
            
            # This logic is for our test run, using the original .flac files
            audio_file = enhanced_audio_dir / f"{signal_id}_unproc.flac" 
            
            if audio_file.exists():
                score = self.score_single_file(audio_file, lyrics)
                raw_scores.append({"signal": signal_id, "whisper": score})
            else:
                print(f"Warning: Could not find audio file at {audio_file}")
        
        valid_scores_df = pd.DataFrame(raw_scores)
        if len(valid_scores_df) == 0:
            print("CRITICAL ERROR: No audio files were found. Check the path in 'test_audio_folder'.")
            return None

        print(f"Computed scores for {len(valid_scores_df)} files.")

        # --- STAGE 2: Calibrate Predictions ---
        print("\n[Stage 2/3] Calibrating predictions using logistic model...")
        
        # Load precomputed scores for the training set (as recommended by baseline)
        train_scores_df = pd.read_json(precomputed_train_scores_path, lines=True)
        train_metadata_df = pd.read_json(train_metadata_path)
        
        # Merge to align scores with true correctness
        train_df = pd.merge(train_scores_df, train_metadata_df, on='signal')
        
        # Fit the logistic model
        model = LogisticModel()
        model.fit(train_df["whisper"], train_df.correctness)
        print("Logistic model fitted on training data.")
        
        # Predict on our validation scores
        valid_scores_df["predicted"] = model.predict(valid_scores_df.whisper)
        
        # --- STAGE 3: Evaluate Final Scores ---
        print("\n[Stage 3/3] Computing final RMSE and NCC scores...")
        
        # Merge our predictions with the ground truth
        final_df = pd.merge(valid_scores_df, metadata_df, on='signal')
        
        scores = compute_scores(final_df["predicted"], final_df["correctness"])
        
        print("\n--- Evaluation Complete! ---")
        print(json.dumps(scores, indent=2))
        return scores


# Example of how you would use this class in your own code:
if __name__ == '__main__':
    # You would run this from your main project folder
    
    # --- THIS IS THE CORRECTED CODE ---
    # We now build the path to the data starting from our reliable, absolute 'project_root'
    cadenza_data_root = project_root / "code" / "audio_files" / "16981327" / "cadenza_data"
    
    # This path points to the original audio, which is correct for our test run.
    test_audio_folder = cadenza_data_root / "valid" / "unprocessed"

    valid_metadata = cadenza_data_root / "metadata" / "valid_metadata.json"
    train_metadata = cadenza_data_root / "metadata" / "train_metadata.json"

    precomputed_train_scores = PATH_TO_BASELINE_CODE / "precomputed" / "cadenza_data.train.whisper.mixture.jsonl"

    # Create the helper
    eval_helper = EvaluationHelper()
    
    # Run the entire evaluation process on the test audio
    final_scores = eval_helper.run_full_evaluation(
        enhanced_audio_dir=test_audio_folder,
        metadata_path=valid_metadata,
        train_metadata_path=train_metadata,
        precomputed_train_scores_path=precomputed_train_scores
    )