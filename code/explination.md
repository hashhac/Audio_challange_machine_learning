Cadenza ICASSP 2026 Challenge: Input/Output and Data Flow GuideThis document details the structure, purpose, and complete input/output flow of the experiment runner for the Cadenza ICASSP 2026 Challenge. The pipeline is designed to facilitate quick development and evaluation of audio enhancement models.Project Structure (Quick Reference)Audio_challange_machine_learning/
├── main.py                     # Orchestrates the full pipeline (Enhancement + Evaluation).
├── compute_csvs.py             # Utility to compare local submission scores.
│
├── code/
│   ├── models/
│   │   └── hearing_aid_models.py # Contains the enhancement model (e.g., NLR1_Model).
│   │
│   ├── evaluation/
│   │   └── evaluation_helper.py  # Wraps the official evaluation logic (The Judge).
│   │
│   └── data_loading.py         # Handles dataset reading and batching (The Librarian).
│
├── clarity/                      # Official Clarity Challenge baseline code (used for evaluation).
│
└── results/
    ├── NLR1/                     # Output folder for enhanced audio files.
    └── submission_NLR1.csv       # Final submission file.
Core Components: Complete Input and Output SpecificationScript/ModuleRoleKey InputsKey Outputsmain.pyThe Orchestrator1. Cadenza Data Root Path (internal config) 2. Enhancement Model (from hearing_aid_models.py)1. Enhanced Audio Folder (results/ModelName/): Processed .wav files. 2. Final Submission CSV (submission_ModelName.csv): Official challenge predictions.code/data_loading.pyThe Librarian1. cadenza_data Parent Directory Path1. PyTorch DataLoader Objects (Batches of data). 2. Batch Dictionary containing {signals: Tensor, metadata: List[Dict], ...}.code/models/hearing_aid_models.pyThe Enhancer1. signal_tensor (PyTorch, stereo audio) 2. audiogram_levels (List of numbers for hearing loss prescription)1. processed_tensor (PyTorch, new enhanced audio signal)code/evaluation/evaluation_helper.pyThe Judge (Scoring Wrapper)1. audio_dir (Path to enhanced .wav files) 2. Metadata Paths (Valid & Train JSON) 3. Precomputed Training Scores (.jsonl)1. Final Submission CSV (Formatted with signal_id and predicted score).compute_csvs.pyThe Comparator1. Multiple Submission CSVs (from project root) 2. Training Set Ground Truth (used for comparison)1. Terminal Output (Comparison table of RMSE and NCC scores).How to Run the Pipeline1. Run a Full Experiment (Enhancement and Official Submission CSV Generation)This is the primary workflow for creating a submission.Configure: In main.py, ensure your model is correctly initialized and enhanced_audio_output_dir is named appropriately (e.g., NLR1 or MyNewModel).Execute:python main.py
This script handles two stages:Stage 1: Processing audio and saving enhanced .wav files to results/.Stage 2: Loading those enhanced files, running the Whisper ASR, calibrating scores using the logistic model, and saving the final submission CSV to the project root.2. Compare Multiple Submission Files (Local Score Comparison)Use this utility to compare the performance of different models against the provided training set ground truth locally.Gather: Ensure all submission .csv files you wish to compare are in the project root directory.Execute:python compute_csvs.py
Note: This comparison uses the training set ground truth and is for relative performance checks only. The final official score comes from the challenge leaderboard.