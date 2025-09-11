import os
import json
import matplotlib.pyplot as plt
class ModelRanker:
    """
    A class to read and plot the results from multiple model evaluations.
    """
    def __init__(self, filenames):
        self.filenames = filenames
        self.all_results = {}

    def load_results(self):
        for filename in self.filenames:
            if not os.path.exists(filename):
                print(f"Warning: File {filename} not found. Skipping.")
                continue
            with open(filename, 'r') as f:
                model_name = os.path.splitext(os.path.basename(filename))[0].replace("results_", "")
                self.all_results[model_name] = json.load(f)

    def plot_results(self):
        if not self.all_results:
            print("No results to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot MSE Loss
        for model_name, results in self.all_results.items():
            epochs = [r["epoch"] for r in results]
            mse_losses = [r["mse_loss"] for r in results]
            ax1.plot(epochs, mse_losses, label=model_name)
        ax1.set_title("MSE Loss vs. Epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("MSE Loss")
        ax1.legend()
        ax1.grid(True)

        # Plot HASPI Score
        for model_name, results in self.all_results.items():
            epochs = [r["epoch"] for r in results]
            haspi_scores = [r["haspi_score"] for r in results]
            ax2.plot(epochs, haspi_scores, label=model_name)
        ax2.set_title("HASPI Score vs. Epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("HASPI Score")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        dir_path = os.path.join(os.path.dirname(__file__), "results", "model_comparison.png")
        if not os.path.exists(os.path.dirname(dir_path)):
            os.makedirs(os.path.dirname(dir_path))
        plt.savefig(dir_path)
        plt.show()
