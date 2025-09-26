# compute_csvs.py

import sys
from pathlib import Path
import pandas as pd
import json
import logging

# --- 1. SETUP: BORROW THE ROBUST PATHING LOGIC ---
# We need to add the baseline code to our path so we can import 'compute_scores'
# This is the same proven setup function from your other scripts.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_paths():
    """Setup all necessary paths to find the baseline's evaluation tools."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent  # Assumes this script is in the project root
    
    baseline_path = project_root / "code" / "audio_files" / "16981327" / "clarity" / "recipes" / "cad_icassp_2026" / "baseline"
    
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline path not found: {baseline_path}")
        
    if str(baseline_path) not in sys.path:
        sys.path.append(str(baseline_path))
        
    logging.info(f"Project root is: {project_root}")
    logging.info(f"Added to Python path: {baseline_path}")
    
    return project_root

# Setup paths before trying to import from the baseline
project_root = setup_paths()

# Now we can safely import the scoring function
from evaluate import compute_scores


# --- 2. THE MAIN LOGIC FOR COMPARING SUBMISSIONS ---

def compare_submission_files(directory_to_scan: Path, ground_truth_metadata_path: Path):
    """
    Finds all .csv files in a directory, treats them as submissions,
    and scores them against a ground truth metadata file.
    """
    logging.info(f"Scanning for submission CSVs in: {directory_to_scan}")
    
    # Use .glob to find all files ending in .csv in the specified directory
    submission_files = list(directory_to_scan.glob("*.csv"))
    
    if not submission_files:
        logging.warning("No .csv submission files found in this directory. Nothing to do.")
        return

    logging.info(f"Found {len(submission_files)} CSV file(s) to evaluate.")
    
    # --- LOAD THE "ANSWER KEY" ---
    # IMPORTANT: The official validation set answers are secret.
    # To get a score for comparison, we use the TRAINING set metadata,
    # as it's the only public data that contains the 'correctness' column.
    # This means the scores are for COMPARING your models, not for the final leaderboard.
    logging.info("Loading ground truth data for comparison...")
    ground_truth_df = pd.read_json(ground_truth_metadata_path)
    # Rename 'signal' to match the submission file's 'signal_ID' for easy merging
    ground_truth_df.rename(columns={'signal': 'signal_ID'}, inplace=True)
    
    # --- LOOP THROUGH AND SCORE EACH FILE ---
    for csv_path in submission_files:
        print("\n" + "="*50)
        print(f"EVALUATING: {csv_path.name}")
        print("="*50)
        
        try:
            # Load the submission file
            submission_df = pd.read_csv(csv_path)
            
            # Merge the predictions with the ground truth answers
            # The 'inner' join keeps only the signals present in BOTH files
            final_df = pd.merge(submission_df, ground_truth_df, on='signal_ID', how="inner")
            
            if len(final_df) == 0:
                print(" -> WARNING: No matching signals found between this CSV and the ground truth data.")
                continue

            # Use the imported 'compute_scores' function to get the results
            scores = compute_scores(final_df["intelligibility_score"], final_df["correctness"])
            
            # Print the results in a clean format
            print(json.dumps(scores, indent=2))
            
        except Exception as e:
            print(f" -> ERROR: Could not process this file. Reason: {e}")


# --- 3. RUN THE SCRIPT ---
if __name__ == '__main__':
    # Define the directory to scan. By default, it's the directory the script is in.
    directory_to_scan = project_root
    
    # Define the path to our "answer key".
    ground_truth_file = project_root / "code" / "audio_files" / "16981327" / "cadenza_data" / "metadata" / "train_metadata.json"
    
    # Run the main function
    compare_submission_files(directory_to_scan, ground_truth_file)