
import torch
import torch.nn as nn
class HearingAidFilterBank(nn.Module):
    """
    A simple PyTorch module to simulate a hearing aid filter bank.
    
    This module uses a series of 1-D convolutional layers, each acting as a
    bandpass filter. The weights of these "filters" are learnable parameters
    that can be optimized during training.
    """
    def __init__(self, n_filters=100, sr=16000, n_taps=128):
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