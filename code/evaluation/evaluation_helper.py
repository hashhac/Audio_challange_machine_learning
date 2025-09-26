# evaluation_helper.py

import sys
from pathlib import Path
import torch
import whisper
import numpy as np
import pandas as pd
import json
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- This pathing logic is now robust and proven to work for your setup ---
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
    if str(baseline_path) not in sys.path:
        sys.path.append(str(baseline_path))
    logging.info(f"Project root is: {project_root}")
    logging.info(f"Added to Python path: {baseline_path}")
    return project_root, baseline_path

project_root, PATH_TO_BASELINE_CODE = setup_paths()

# Now we can import their tools directly
from transcription_scorer import SentenceScorer
from shared_predict_utils import LogisticModel

class EvaluationHelper:
    """
    A class to perform the full end-to-end evaluation pipeline for the
    Cadenza ICASSP 2026 challenge.
    """
    def __init__(self, whisper_version="base.en"):
        print("Initializing Evaluation Helper...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load the Whisper model and the Scorer on initialization
        self.asr_model = whisper.load_model(whisper_version, device=device)
        contractions_file = PATH_TO_BASELINE_CODE / "contractions.csv"
        self.scorer = SentenceScorer(str(contractions_file))
        print("Initialization complete.")

    def _score_single_file(self, audio_path: Path, reference_lyrics: str) -> float:
        """[Internal] Computes the raw Whisper correctness score for one audio file."""
        hypothesis = self.asr_model.transcribe(
            str(audio_path), fp16=False, language="en", temperature=0.0
        )["text"]
        results = self.scorer.score([reference_lyrics], [hypothesis])
        total_words = results.substitutions + results.deletions + results.hits
        return 0.0 if total_words == 0 else results.hits / total_words

    def evaluate_audio_folder(self, 
                              audio_dir: Path, 
                              metadata_path: Path, 
                              train_metadata_path: Path,
                              precomputed_train_scores_path: Path,
                              output_csv_path: Path):
        """
        Runs the full 3-stage evaluation pipeline on a folder of audio files
        and produces the final submission CSV.
        """
        print(f"--- Starting Full Evaluation on folder: {audio_dir} ---")
        
        # --- STAGE 1: Compute Raw Scores from Audio ---
        print("\n[Stage 1/3] Computing raw Whisper scores for your audio...")
        metadata_df = pd.read_json(metadata_path)
        raw_scores = []
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Scoring Audio"):
            signal_id = row['signal']
            lyrics = row['prompt']
            
            # This logic must handle both original .flac and your enhanced .wav files
            audio_file_flac = audio_dir / f"{signal_id}_unproc.flac"
            audio_file_wav = audio_dir / f"{signal_id}.wav"
            
            audio_to_process = None
            if audio_file_wav.exists():
                audio_to_process = audio_file_wav
            elif audio_file_flac.exists():
                audio_to_process = audio_file_flac
            
            if audio_to_process:
                score = self._score_single_file(audio_to_process, lyrics)
                raw_scores.append({"signal": signal_id, "whisper.mixture": score})
            else:
                print(f"Warning: Could not find audio for {signal_id} (looked for .wav and .flac)")
        
        valid_scores_df = pd.DataFrame(raw_scores)
        print(f"Computed scores for {len(valid_scores_df)} files.")

        # --- STAGE 2: Calibrate Predictions ---
        print("\n[Stage 2/3] Calibrating predictions using logistic model...")
        data_list_train = [json.loads(line) for line in open(precomputed_train_scores_path, 'r', encoding='utf-8')]
        train_scores_df = pd.DataFrame(data_list_train)
        train_metadata_df = pd.read_json(train_metadata_path)
        train_df = pd.merge(train_scores_df, train_metadata_df, on='signal')
        
        model = LogisticModel()
        model.fit(train_df["whisper.mixture"], train_df.correctness)
        print("Logistic model fitted on training data.")
        
        # --- STAGE 3: Save Final Submission File ---
        print("\n[Stage 3/3] Generating and saving submission file...")
        valid_scores_df["predicted"] = model.predict(valid_scores_df["whisper.mixture"])
        
        submission_df = valid_scores_df[["signal", "predicted"]]
        submission_df.columns = ["signal_ID", "intelligibility_score"]
        submission_df.to_csv(output_csv_path, index=False)
        
        print("\n--- Evaluation Complete! ---")
        print(f"Official submission file saved to: {output_csv_path}")
        return output_csv_path


if __name__ == '__main__':
    # --- 1. DEFINE ALL YOUR PATHS ---
    cadenza_data_root = project_root / "code" / "audio_files" / "16981327" / "cadenza_data"
    
    # Path to the audio you want to evaluate.
    # To get a benchmark, point this to the original unprocessed audio.
    # LATER, you will change this to point to your enhanced audio folder.
    audio_to_evaluate_folder = cadenza_data_root / "valid" / "unprocessed"
    # audio_to_evaluate_folder = project_root / "results" / "my_enhanced_audio" # <-- Your goal is to use this one
    
    # Paths to the metadata and precomputed score files
    valid_metadata = cadenza_data_root / "metadata" / "valid_metadata.json"
    train_metadata = cadenza_data_root / "metadata" / "train_metadata.json"
    precomputed_train_scores = PATH_TO_BASELINE_CODE / "precomputed" / "cadenza_data.train.whisper.jsonl"
    
    # Path for the final output file
    output_csv = project_root / "my_final_submission.csv"

    # --- 2. CREATE AND RUN THE EVALUATOR ---
    eval_helper = EvaluationHelper() 
    
    eval_helper.evaluate_audio_folder(
        audio_dir=audio_to_evaluate_folder,
        metadata_path=valid_metadata,
        train_metadata_path=train_metadata,
        precomputed_train_scores_path=precomputed_train_scores,
        output_csv_path=output_csv
    )