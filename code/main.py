from pathlib import Path
import os
import sys
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
from clarity.enhancer.nalr import NALR
from project_setup import setup_project_directories

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
        print(f"pyclarity version: {NALR.__repr__=}")
    except Exception as e:
        print(f"An error occurred during imports: {e}")

if __name__ == "__main__":
    load_dotenv()
    setup_project_directories()
    big_print_of_imports()    

