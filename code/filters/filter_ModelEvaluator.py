import torch
import torch.nn as nn
import numpy as np
import json
import os
from clarity.evaluator.haspi import haspi_v2
#TODO ADD JSON PATH SETUP need to pull orginal working dir with os then pull filter path from project setup to then specify json save location
class ModelEvaluator:
    """
    A class to train and evaluate a single model, saving results to a JSON file.
    """
    def __init__(self, model, model_name, target_signal, input_signal, audiogram, sr, epochs=50, lr=0.01, optimizer=torch.optim.Adam, loss_fn=nn.MSELoss):
        self.model = model
        self.model_name = model_name
        self.target_signal = target_signal
        self.input_signal = input_signal
        self.audiogram = audiogram
        self.sr = sr
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)
        self.loss_fn = loss_fn()
        self.results = []
        self.filename = f"results_{model_name}.json"

    def evaluate(self):
        print(f"Starting calibration for {self.model_name}...")
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            
            # Forward pass: process the signal with the hearing aid model
            output_signal = self.model(self.input_signal)
            
            # Sanity check: ensure the output shape is correct before calculating loss
            if output_signal.shape[1] != self.target_signal.shape[1]:
                print("Warning: Output signal shape mismatch. Slicing to match target.")
                output_signal = output_signal[:, :self.target_signal.shape[1]]
                
            # Calculate the differentiable MSE loss
            loss = self.loss_fn(output_signal, self.target_signal)
            
            # Backward pass: compute gradients and update parameters
            loss.backward()
            self.optimizer.step()
            
            # --- Evaluation with HASPI (not used for training) ---
            # Squeeze, detach, and convert to numpy.
            processed_signal_np = output_signal.squeeze().detach().numpy()
            
            # Another shape check for HASPI
            if processed_signal_np.shape != self.target_signal.squeeze().numpy().shape:
                 processed_signal_np = processed_signal_np[:self.target_signal.squeeze().numpy().shape[0]]

            haspi_score_tuple = haspi_v2(
                reference=self.target_signal.squeeze().numpy(),
                reference_sample_rate=self.sr,
                processed=processed_signal_np,
                processed_sample_rate=self.sr,
                audiogram=self.audiogram
            )
            
            haspi_score = haspi_score_tuple[0]
            
            # Store the results
            self.results.append({
                "epoch": epoch + 1,
                "mse_loss": loss.item(),
                "haspi_score": haspi_score
            })
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, MSE Loss: {loss.item():.6f}, HASPI Score: {haspi_score:.4f}")
                
        print(f"Calibration for {self.model_name} finished.")
        self.save_results()

    def save_results(self):
        with open(self.filename, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {self.filename}")