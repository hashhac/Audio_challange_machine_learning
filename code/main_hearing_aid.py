# main.py (or run_experiment.py) - The Final, Corrected Version

from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import logging
import sys

# --- Imports and Path Setup (This part is already correct) ---
project_root = Path(__file__).resolve().parent.parent
code_path = project_root / "code"
if str(code_path) not in sys.path:
    sys.path.insert(0, str(code_path))

from models.hearing_aid_models import NLR1_Model, enhance_and_save_dataset
from data_loading import create_dataloaders
from evaluation.evaluation_helper import EvaluationHelper, setup_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- WRAP THE ENTIRE PROCESS IN A MAIN FUNCTION ---
def run_full_experiment():
    """
    Orchestrates the entire process:
    1. Sets up all necessary paths.
    2. Runs the enhancement model to generate audio files.
    3. Runs the evaluation helper to score the generated files.
    """
    
    # --- SETUP (This part is already correct) ---
    logging.info("Setting up paths for the experiment...")
    project_root, baseline_code_path = setup_paths()
    cadenza_data_root = project_root / "code" / "audio_files" / "16981327" / "cadenza_data"
    enhanced_audio_output_dir = project_root / "results" / "my_enhanced_audio"
    enhanced_audio_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Original data is at: {cadenza_data_root}")
    logging.info(f"Enhanced audio will be saved to: {enhanced_audio_output_dir}")

    # --- PHASE 1: ENHANCEMENT --- 
    logging.info("--- Starting Phase 1: Audio Enhancement ---")

    # Create dataloaders (keep same call so paths / splits remain identical)
    _, valid_loader = create_dataloaders(cadenza_data_root.parent, batch_size=4)

    # Use the already-tested helper in hearing_aid_models.py so processing matches the file-local test-run.
    # This replaces the manual loop and guarantees the same processing & save behavior.
    enhance_and_save_dataset(valid_loader, enhanced_audio_output_dir, debug_pairs=0)

    logging.info(f"--- Phase 1 Complete: All enhanced audio saved. ---")

    # --- PHASE 2: EVALUATION (This part is already correct) ---
    logging.info("--- Starting Phase 2: Evaluation ---")
    
    valid_metadata = cadenza_data_root / "metadata" / "valid_metadata.json"
    train_metadata = cadenza_data_root / "metadata" / "train_metadata.json"
    precomputed_train_scores = baseline_code_path / "precomputed" / "cadenza_data.train.whisper.jsonl"
    output_csv = project_root / f"submission_{enhanced_audio_output_dir.name}.csv"

    eval_helper = EvaluationHelper()
    
    eval_helper.evaluate_audio_folder(
        audio_dir=enhanced_audio_output_dir,
        metadata_path=valid_metadata,
        train_metadata_path=train_metadata,
        precomputed_train_scores_path=precomputed_train_scores,
        output_csv_path=output_csv
    )
    
    logging.info(f"--- Phase 2 Complete: Experiment Finished! ---")


# --- THIS MAKES THE SCRIPT RUNNABLE ---
if __name__ == '__main__':
    run_full_experiment()