import numpy as np
import torch
import os
from filter_HearingAidFilterBank import HearingAidFilterBank
from filter_ModelEvaluator import ModelEvaluator
from filter_ModelRanker import ModelRanker
from clarity.utils.audiogram import Audiogram


if __name__ == "__main__":
    # Parameters for the simulation
    # check if json already exists
    n_samples = 16000 # 1 second of audio
    sr = 16000
    
    # Generate a complex, realistic dummy signal for demonstration
    t = np.linspace(0, 1, n_samples, endpoint=False)
    target_signal_np = np.sum([np.sin(2 * np.pi * f * t) for f in [200, 500, 1000, 2000, 4000]], axis=0)
    target_signal_np /= np.max(np.abs(target_signal_np)) # Normalize
    target_signal_torch = torch.from_numpy(target_signal_np).float().unsqueeze(0)
    
    # Noisy input signal with a more significant noise component
    noise = np.random.randn(n_samples) * 0.5
    input_signal_np = target_signal_np + noise
    input_signal = torch.from_numpy(input_signal_np).float().unsqueeze(0).unsqueeze(0)
    
    # Define the audiogram for HASPI evaluation
    frequencies = np.array([250, 500, 1000, 2000, 4000, 6000])
    levels = np.array([20, 25, 30, 40, 50, 60])
    mock_audiogram = Audiogram(frequencies=frequencies, levels=levels)

    # Define the models to be evaluated
    models_to_test = {
        "Model_A_100_Filters": HearingAidFilterBank(n_filters=100),
        "Model_B_50_Filters": HearingAidFilterBank(n_filters=50),
        "Model_C_200_Taps": HearingAidFilterBank(n_taps=200),
    }
    #TODO add more models as needed to evaluate a moving band bass filter and a model with non linearities, also a model with more layers etc 

    results_files = []
    # Evaluate each model
    for name, model_instance in models_to_test.items():
        dir_path = os.path.join(os.path.dirname(__file__), "results", f"results_{name}.json")
        if os.path.exists(dir_path):
            print(f"Results for {name} already exist. Skipping evaluation.")
            results_files.append(dir_path)
            continue
        evaluator = ModelEvaluator(
            model=model_instance,
            model_name=name,
            target_signal=target_signal_torch,
            input_signal=input_signal,
            audiogram=mock_audiogram,
            sr=sr
        )
        evaluator.evaluate()
        results_files.append(evaluator.filename)

    # Rank and plot the results
    ranker = ModelRanker(results_files)
    ranker.load_results()
    ranker.plot_results()