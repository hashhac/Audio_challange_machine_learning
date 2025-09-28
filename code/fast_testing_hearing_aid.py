# run_fast_experiment.py

from pathlib import Path
import logging
import sys
import random
from torch.utils.data import DataLoader, Subset

# --- Imports and Path Setup (This part is already correct) ---
project_root = Path(__file__).resolve().parent.parent
code_path = project_root / "code"
if str(code_path) not in sys.path:
    sys.path.insert(0, str(code_path))

from models.hearing_aid_models import enhance_and_save_dataset
from data_loading import create_dataloaders
from evaluation.evaluation_helper import EvaluationHelper, setup_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- WRAP THE ENTIRE PROCESS IN A MAIN FUNCTION ---
def run_fast_training_evaluation(num_samples: int = 400):
    """
    Orchestrates the entire pipeline on a small, random subset of the
    TRAINING data for fast iteration and model comparison.
    """
    
    # --- SETUP ---
    logging.info(f"--- Starting FAST evaluation on {num_samples} random training samples ---")
    
    project_root, baseline_code_path = setup_paths()
    cadenza_data_root = project_root / "code" / "audio_files" / "16981327" / "cadenza_data"
    
    # Give the output a descriptive name
    model_name = f"my_enhanced_audio_TRAIN_{num_samples}_samples"
    enhanced_audio_output_dir = project_root / "results" / model_name
    enhanced_audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Enhanced audio will be saved to: {enhanced_audio_output_dir}")

    # --- PHASE 1: ENHANCEMENT (on a random subset) --- 
    logging.info("--- Starting Phase 1: Audio Enhancement ---")

    # First, create a dataloader for the FULL training set
    train_loader_full, _ = create_dataloaders(cadenza_data_root.parent, batch_size=4)

    # --- THE SUBSET TRICK ---
    # 1. Get the total number of items in the dataset
    num_total_samples = len(train_loader_full.dataset)
    # 2. Create a list of random indices to select
    random_indices = random.sample(range(num_total_samples), num_samples)
    # 3. Create a "Subset" dataset, which is a lightweight wrapper
    subset_dataset = Subset(train_loader_full.dataset, random_indices)
    # 4. Create a new DataLoader that will only yield data from our random subset
    fast_train_loader = DataLoader(subset_dataset, batch_size=4, collate_fn=train_loader_full.collate_fn)
    # -------------------------

    # Use our proven worker function, but with the fast dataloader
    enhance_and_save_dataset(fast_train_loader, enhanced_audio_output_dir)

    logging.info(f"--- Phase 1 Complete: {num_samples} enhanced audio files saved. ---")

    # --- PHASE 2: EVALUATION ---
    logging.info("--- Starting Phase 2: Evaluation ---")
    
    train_metadata = cadenza_data_root / "metadata" / "train_metadata.json"
    precomputed_train_scores = baseline_code_path / "precomputed" / "cadenza_data.train.whisper.jsonl"
    
    # Give the output CSV a matching descriptive name
    output_csv = project_root / f"submission_train_ENHANCED_{num_samples}_samples.csv"

    eval_helper = EvaluationHelper()
    
    # Point the evaluator at our new folder of 400 enhanced files
    eval_helper.evaluate_audio_folder(
        audio_dir=enhanced_audio_output_dir,
        metadata_path=train_metadata, # The evaluator still needs the full metadata to find the prompts
        train_metadata_path=train_metadata,
        precomputed_train_scores_path=precomputed_train_scores,
        output_csv_path=output_csv
    )
    
    logging.info(f"--- Fast Experiment Finished! ---")
    logging.info(f"Created submission file: {output_csv.name}")
    logging.info("You can now run 'evaluate_csv.py' to get the score for this subset.")


# --- THIS MAKES THE SCRIPT RUNNABLE ---
if __name__ == '__main__':
    run_fast_training_evaluation(num_samples=400)