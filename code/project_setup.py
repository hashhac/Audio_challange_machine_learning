import os
import sys
from pathlib import Path

def setup_project_directories():
    """
    Creates the necessary project directories if they don't exist
    and returns a dictionary of their paths.
    """
    work_dir = Path(__file__).parent
    
    paths = {
        "save_path": work_dir / "saved_models",
        "audio_path": work_dir / "audio_files",
        "audio_data": work_dir / "audio_files" / "data",
        "audio_validation": work_dir / "audio_files" / "validation",
        "neural_networks_path": work_dir / "neural_networks",
        "filter_path": work_dir / "filters",
        "predictive_models_path": work_dir / "predictive_models",
        "plotting_path": work_dir / "plots",
        "json_path": work_dir / "json_data",
        "evaluation_path": work_dir / "evaluation",
    }

    # Create each directory if it doesn't exist
    for path in paths.values():
        if not path.exists():
            print(f"Creating directory: {path}")
            os.makedirs(path)
            
    # Add all directories to the system path
    for path in paths.values():
        if str(path) not in sys.path:
            sys.path.append(str(path))
            
    return paths