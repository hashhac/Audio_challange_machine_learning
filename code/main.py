from pathlib import Path
import os
import time 
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import sklearn
import random
import librosa.display
import noisereduce as nr
from dotenv import load_dotenv
from numpy import ndarray
from typing import List, Tuple, Dict, Any, Union
from dataclasses import dataclass
import transformers


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function {func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@measure_time
def big_print_of_imports():
    try:
        print("All imports are working correctly.")
        print(f"torch version: {torch.__version__=}")
        print(f"torchaudio version: {torchaudio.__version__=}")
        print(f"librosa version: {librosa.__version__=}")
        print(f"numpy version: {np.__version__=}")
        print(f"soundfile version: {sf.__version__=}")
        print(f"matplotlib version: {matplotlib.__version__=}")
        print(f"pandas version: {pd.__version__=}")
        print(f"seaborn version: {sns.__version__=}")
        print(f"sklearn version: {sklearn.__version__=}")
        print(f"transformers version: {transformers.__version__=}")
    except Exception as e:
        print(f"An error occurred during imports: {e}")


def dir_check():
    work_dir = Path(__file__).parent
    save_path = work_dir / "saved_models"
    audio_path = work_dir / "audio_files"
    audio_data =audio_path / "data"
    audio_validation = audio_path / "validation"
    neural_networks_path = work_dir / "neural_networks"
    filter_path = work_dir / "filters"
    predictive_models_path = work_dir / "predictive_models"

    file_paths_list = [work_dir, save_path, audio_path, audio_data, audio_validation, neural_networks_path, filter_path, predictive_models_path]
    for path in file_paths_list:
        if not path.exists():
            os.makedirs(path)

if __name__ == "__main__":
    load_dotenv()
    dir_check()
    big_print_of_imports()

