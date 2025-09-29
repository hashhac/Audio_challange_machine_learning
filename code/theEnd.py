# the_end.py
# this was because the fast nearal eval had a pathing error and i didnt want to fix it as i am so close to being done only needing to run eval
# so this script just runs the eval on the folder created by fast_nearal_eval.py
from pathlib import Path
import logging
import sys

# --- 1. SETUP: ROBUST PATHING AND IMPORTS ---

def setup_paths():
    """
    Finds the project root by searching upwards from the current file for a known directory.
    This is the most robust way to handle pathing.
    """
    current_file = Path(__file__).resolve()
    project_root = current_file
    # We will search upwards for a reliable anchor, the main project folder name.
    # MAKE SURE 'Audio_challange_machine_learning' is the correct name of your top-level folder.
    while project_root.name != 'Audio_challange_machine_learning' and project_root.parent != project_root:
        project_root = project_root.parent
        
    if project_root.name != 'Audio_challange_machine_learning':
        raise FileNotFoundError("Could not find the project root 'Audio_challange_machine_learning'.")

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
from evaluation.evaluation_helper import EvaluationHelper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- WRAP THE ENTIRE PROCESS IN A MAIN FUNCTION ---
def get_final_score_from_existing_audio():
    """
    This script's only job is to evaluate an EXISTING folder of enhanced audio
    and produce the final submission CSV. It does no audio processing.
    """
    
    # --- SETUP ---
    logging.info("--- Starting Final Evaluation ---")
    
    cadenza_data_root = project_root / "code" / "audio_files" / "16981327" / "cadenza_data"
    
    # --- THIS IS THE CRITICAL PATH ---
    # We point this DIRECTLY to the folder of audio you successfully created.
    audio_folder_to_evaluate = project_root / "results" / "FINAL_NEURAL_RUN_400_samples"
    
    if not audio_folder_to_evaluate.exists():
        logging.error(f"CRITICAL: The audio folder was not found at {audio_folder_to_evaluate}")
        logging.error("Please ensure you have run the neural enhancement script first.")
        return

    logging.info(f"Found {len(list(audio_folder_to_evaluate.glob('*.wav')))} audio files to evaluate in:")
    logging.info(f" -> {audio_folder_to_evaluate}")
    # ------------------------------------

    # --- PHASE 1: SKIPPED --- 
    logging.info("--- Skipping Audio Enhancement Phase ---")

    # --- PHASE 2: EVALUATION ---
    logging.info("--- Starting Phase 2: Evaluation ---")
    
    train_metadata = cadenza_data_root / "metadata" / "train_metadata.json"
    precomputed_train_scores = baseline_code_path / "precomputed" / "cadenza_data.train.whisper.jsonl"
    
    # Give the output CSV a final, descriptive name
    output_csv = project_root / "submission_train_NEURAL_NETWORK_400_samples.csv"

    eval_helper = EvaluationHelper()
    
    # Point the evaluator at our folder of 400 enhanced files
    eval_helper.evaluate_audio_folder(
        audio_dir=audio_folder_to_evaluate,
        metadata_path=train_metadata,
        train_metadata_path=train_metadata,
        precomputed_train_scores_path=precomputed_train_scores,
        output_csv_path=output_csv
    )
    
    logging.info(f"--- Final CSV Created! ---")
    logging.info(f"File created: {output_csv.name}")
    logging.info("You can now run 'evaluate_csv.py' to get your final score and be free.")


# --- THIS MAKES THE SCRIPT RUNNABLE ---
if __name__ == '__main__':
    get_final_score_from_existing_audio()