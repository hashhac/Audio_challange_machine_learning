import torch
import torch.nn as nn
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# We now import the necessary components from pyclarity
# The monaural version of HASPI, which is appropriate for a single-channel model.
from clarity.evaluator.haspi import haspi_v2
from clarity.utils.audiogram import Audiogram

# --- Part 1: PyTorch Filter Bank Module ---
class HearingAidFilterBank(nn.Module):
    """
    A simple PyTorch module to simulate a hearing aid filter bank.
    
    This module uses a series of 1-D convolutional layers, each acting as a
    bandpass filter. The weights of these "filters" are learnable parameters
    that can be optimized during training.
    """
    def __init__(self, n_filters=80, sr=16000, n_taps=128):
        """
        Initializes the filter bank.
        
        Args:
            n_filters (int): The number of independent frequency bands (filters).
            sr (int): The sample rate of the audio signal.
            n_taps (int): The number of taps for each FIR filter.
        """
        super().__init__()
        self.n_filters = n_filters
        self.sr = sr
        self.n_taps = n_taps
        
        # We use Conv1d to implement the filter bank. Each output channel
        # corresponds to a single filter. The gain of each filter will be
        # determined by the a learnable gain parameter.
        # We initialize with a simple impulse response to start.
        self.filters = nn.Conv1d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=n_taps,
            padding=n_taps // 2,
            bias=False
        )

        # Initialize filter weights as a simple delta function
        # A more realistic implementation would use pre-designed filters,
        # for example, from a gammatone filter bank.
        with torch.no_grad():
            self.filters.weight.zero_()
            self.filters.weight[:, :, self.n_taps // 2] = 1.0

        # The issue was here: The shape of the gains tensor caused incorrect broadcasting.
        # We need a shape that is compatible with element-wise multiplication with
        # the sub-bands tensor, so it should be (batch_size, n_filters, 1).
        # We use a shape of (1, n_filters, 1) and let PyTorch broadcast across the batch dimension.
        self.gains = nn.Parameter(torch.ones(1, n_filters, 1))

    def forward(self, x):
        """
        Applies the filter bank and gains to the input signal.
        
        Args:
            x (torch.Tensor): The input audio signal, with shape (batch, 1, samples).
            
        Returns:
            torch.Tensor: The enhanced audio signal.
        """
        # Apply the filters to get sub-band signals.
        # Shape: (batch_size, n_filters, n_samples_padded)
        sub_bands = self.filters(x)
        
        # Apply the learnable gains to each sub-band.
        # Shape: (batch_size, n_filters, n_samples_padded)
        processed_sub_bands = sub_bands * self.gains
        
        # Sum the processed sub-bands to reconstruct the signal.
        # This will collapse the channel dimension (dim=1).
        # We then squeeze to ensure the output is a 2D tensor (batch, samples).
        # Shape: (batch_size, n_samples_padded)
        output = torch.sum(processed_sub_bands, dim=1)
        
        # The Conv1d layer with padding adds one sample. We must slice the
        # output to match the input signal's length exactly.
        # Shape: (batch_size, n_samples)
        output = output[:, :x.shape[2]]
        
        return output

# --- Part 2: Mock Training Loop (Calibration) ---
def mock_training_loop():
    """
    Demonstrates a simple "calibration" loop using PyTorch, now with HASPI.
    
    In a real-world scenario, you would use data from the Clarity Project
    and a more sophisticated loss function that accounts for a listener's
    audiogram and the target speech signal.
    """
    # Parameters for the simulation
    n_filters = 100
    sr = 16000
    n_samples = 16000 # 1 second of audio
    
    # 1. Initialize the model and loss function
    model = HearingAidFilterBank(n_filters=n_filters, sr=sr)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # We will use Mean Squared Error (MSE) as our differentiable loss function
    loss_fn = nn.MSELoss()

    # 2. Generate a more complex, realistic dummy signal for demonstration
    # This will result in a more realistic initial HASPI score.
    
    # A more complex target signal with multiple frequencies
    t = np.linspace(0, 1, n_samples, endpoint=False)
    target_signal_np = np.sum([np.sin(2 * np.pi * f * t) for f in [200, 500, 1000, 2000, 4000]], axis=0)
    target_signal_np /= np.max(np.abs(target_signal_np)) # Normalize
    target_signal_torch = torch.from_numpy(target_signal_np).float().unsqueeze(0)
    
    # Noisy input signal with a more significant noise component
    noise = np.random.randn(n_samples) * 0.5
    input_signal_np = target_signal_np + noise
    input_signal = torch.from_numpy(input_signal_np).float().unsqueeze(0).unsqueeze(0)
    
    # 3. Define the audiogram for HASPI evaluation
    frequencies = np.array([250, 500, 1000, 2000, 4000, 6000])
    levels = np.array([20, 25, 30, 40, 50, 60])
    mock_audiogram = Audiogram(frequencies=frequencies, levels=levels)

    # 4. Training/Calibration Loop
    epochs = 20
    print("Starting mock calibration loop with HASPI...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass: process the signal with the hearing aid model
        output_signal = model(input_signal)
        
        # Calculate the differentiable MSE loss
        loss = loss_fn(output_signal, target_signal_torch)
        
        # Backward pass: compute gradients and update parameters
        loss.backward()
        optimizer.step()
        
        # --- Evaluation with HASPI (not used for training) ---
        # Squeeze, detach, and convert to numpy.
        processed_signal_np = output_signal.squeeze().detach().numpy()
        
        # Make sure the shapes are correct before passing to haspi_v2
        if processed_signal_np.shape != target_signal_np.shape:
             # The forward pass now correctly produces a 1D tensor, but we need
             # to ensure its length is correct.
             processed_signal_np = processed_signal_np[:n_samples]
        
        haspi_score_tuple = haspi_v2(
            reference=target_signal_np,
            reference_sample_rate=sr,
            processed=processed_signal_np,
            processed_sample_rate=sr,
            audiogram=mock_audiogram
        )
        
        haspi_score = haspi_score_tuple[0]

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs}, MSE Loss: {loss.item():.6f}, HASPI Score: {haspi_score:.4f}")
            
    print("Calibration finished.")
    
    # 5. Show results
    with torch.no_grad():
        final_output = model(input_signal).squeeze().numpy()
        
    print("Original input and final output signal shapes:")
    print(f"Input: {input_signal_np.shape}")
    print(f"Output: {final_output.shape}")

    # Plotting for visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(target_signal_np)
    plt.title("Target Signal (Clean)")
    
    plt.subplot(3, 1, 2)
    plt.plot(input_signal_np)
    plt.title("Input Signal (Noisy)")
    
    plt.subplot(3, 1, 3)
    plt.plot(processed_signal_np)
    plt.title("Output Signal (After 'Calibration' with MSE)")
    plt.tight_layout()
    plt.show()
    
# Run the simulation
if __name__ == "__main__":
    # Note: You will need to install the pyclarity library to run this code.
    # pip install pyclarity
    mock_training_loop()
