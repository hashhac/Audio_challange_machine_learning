# prepare_submission.py

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_submission_file(
    source_csv_path: Path, 
    final_submission_path: Path
):
    """
    Reads a standard CSV with headers and saves it in the strict EvalAI format.
    """
    if not source_csv_path.exists():
        logging.error(f"Source file not found: {source_csv_path}")
        return

    logging.info(f"Reading predictions from: {source_csv_path}")
    
    # Read the CSV you've already created
    submission_df = pd.read_csv(source_csv_path)
    
    # --- THIS IS THE MAGIC ---
    # Save the DataFrame to a new CSV file with two crucial options:
    # 1. header=False: This tells pandas not to write the column names.
    # 2. index=False: This tells pandas not to write the row numbers (0, 1, 2...).
    submission_df.to_csv(final_submission_path, header=False, index=False)
    # -------------------------
    
    logging.info(f"Success! Your official submission file is ready at: {final_submission_path}")


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent
    
    # Define the input file (the one your main.py script created)
    # MAKE SURE THIS FILENAME IS CORRECT
    source_file = project_root / "submission_my_enhanced_audio.csv"
    
    # Define the final output file (this is the one you will upload)
    final_file = project_root / "submission_for_evalai.csv"
    
    create_submission_file(source_file, final_file)