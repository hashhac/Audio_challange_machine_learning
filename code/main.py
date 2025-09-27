# run_experiment.py

from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import logging
import sys

# Ensure the project's `code/` directory is on sys.path so we can import modules
# like `models.hearing_aid_models` and `evaluation.evaluation_helper` without
# requiring `code` to be a package. This makes imports reliable when running
# the script from the project root or other working directories.
project_root = Path(__file__).resolve().parent
# `main.py` lives in the `code/` folder; the project root is its parent
project_root = project_root.parent
code_path = project_root / "code"
if str(code_path) not in sys.path:
    sys.path.insert(0, str(code_path))

# --- 1. IMPORT YOUR CUSTOM-BUILT MODULES ---
# Import the model and evaluation helpers from inside the code/ tree
from models.hearing_aid_models import NLR1_Model, enhance_and_save_dataset
from data_loading import create_dataloaders
from evaluation.evaluation_helper import EvaluationHelper, setup_paths

# Configure logging for a clean output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- WRAP THE ENTIRE PROCESS IN A MAIN FUNCTION ---
def run_full_experiment():
    """
    Orchestrates the entire process:
    1. Sets up all necessary paths.
    2. Runs the enhancement model to generate audio files.
    3. Runs the evaluation helper to score the generated files.
    """
    
    # --- SETUP ---
    logging.info("Setting up paths for the experiment...")
    
    # Use the robust setup_paths function to get our project root
    project_root, baseline_code_path = setup_paths()
    
    # Define the location of the original Cadenza data
    cadenza_data_root = project_root / "code" / "audio_files" / "16981327" / "cadenza_data"
    
    # Define the output folder for your model's enhanced audio.
    # This is the critical folder that connects the two phases.
    enhanced_audio_output_dir = project_root / "results" / "my_enhanced_audio"
    enhanced_audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Original data is at: {cadenza_data_root}")
    logging.info(f"Enhanced audio will be saved to: {enhanced_audio_output_dir}")

    # --- PHASE 1: ENHANCEMENT ---
    # In this phase, we use your BaselineModel to process the validation set
    # and save the output audio files.
    
    logging.info("--- Starting Phase 1: Audio Enhancement ---")
    model = NLR1_Model()
    _, valid_loader = create_dataloaders(cadenza_data_root.parent, batch_size=4)

    # Loop through the validation data, process it, and save it
    for batch in tqdm(valid_loader, desc="Enhancing Audio Files"):
        signals = batch['signal']
        metadata_list = batch['metadata']

        for i in range(len(signals)):
            metadata = metadata_list[i]
            signal_id = metadata['signal']

            # --- THIS IS THE CORRECTED LOGIC ---
            # We now use the exact same, proven method as your test script.
            # 1. Get the hearing loss label from the raw metadata.
            hearing_loss_label = metadata.get('hearing_loss', "No Loss")
            
            # 2. Look up the correct audiogram from the model's default dictionary.
            audiogram = DEFAULT_AUDIOGRAMS.get(hearing_loss_label, DEFAULT_AUDIOGRAMS["No Loss"])

            # 3. Call the model with the DIRECT list of numbers.
            processed_signal_tensor = model.process_audio(signals[i], audiogram_levels=audiogram)
            # ------------------------------------
            
            output_filename = enhanced_audio_output_dir / f"{signal_id}.wav"
            processed_signal_numpy = processed_signal_tensor.cpu().numpy().T
            sf.write(output_filename, processed_signal_numpy, model.sample_rate)

    logging.info(f"--- Phase 1 Complete: All enhanced audio saved. ---")

    # --- PHASE 2: EVALUATION ---
    # Now that you have a folder full of your enhanced .wav files,
    # we use the EvaluationHelper to "judge" them and create the submission file.
    
    logging.info("--- Starting Phase 2: Evaluation ---")
    
    # Define paths needed by the evaluation helper
    valid_metadata = cadenza_data_root / "metadata" / "valid_metadata.json"
    train_metadata = cadenza_data_root / "metadata" / "train_metadata.json"
    precomputed_train_scores = baseline_code_path / "precomputed" / "cadenza_data.train.whisper.jsonl"
    output_csv = project_root / f"submission_{enhanced_audio_output_dir.name}.csv"

    # Initialize the evaluation helper
    eval_helper = EvaluationHelper()
    
    # Call the main evaluation function, pointing it to your NEW audio folder
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