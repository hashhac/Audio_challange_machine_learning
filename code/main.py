# run_experiment.py

from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import logging

# --- 1. IMPORT YOUR CUSTOM-BUILT MODULES ---
# We are now importing the two major components you have built and tested:
# - The model that ENHANCES the audio (from base_line.py)
# - The helper that EVALUATES the audio (from evaluation_helper.py)
from code.base_line import BaselineModel, create_dataloaders
from code.evaluation.evaluation_helper import EvaluationHelper, setup_paths

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
    
    # Initialize your hearing aid model
    model = BaselineModel()
    
    # Create a dataloader for the validation set (the audio we need to process)
    # We pass the parent of 'cadenza_data' to your create_dataloaders function
    _, valid_loader = create_dataloaders(cadenza_data_root.parent, batch_size=4) # Smaller batch size for processing

    # Loop through the validation data, process it, and save it
    for batch in tqdm(valid_loader, desc="Enhancing Audio Files"):
        signals = batch['signal']
        metadata_list = batch['metadata']
        
        for i in range(len(signals)):
            metadata = metadata_list[i]
            audiogram_levels = metadata['audiogram_levels_l']
            
            # Use your model to process the audio in memory
            processed_signal_tensor = model.process_audio(signals[i], audiogram_levels)
            
            # Define the output filename based on the signal ID
            signal_id = metadata['signal']
            output_filename = enhanced_audio_output_dir / f"{signal_id}.wav"
            
            # Convert the PyTorch tensor to a NumPy array that soundfile can use
            # .cpu() moves data from GPU to CPU (if applicable)
            # .numpy() converts to NumPy format
            # .T transposes the shape from (channels, samples) to (samples, channels)
            processed_signal_numpy = processed_signal_tensor.cpu().numpy().T
            
            # Save the processed audio as a .wav file
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