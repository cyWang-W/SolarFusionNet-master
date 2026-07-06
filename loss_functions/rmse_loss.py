from torch import nn
import torch
import torch.nn.functional as F
import warnings


class ComplexRMSELoss(nn.Module):
    def __init__(self):
        super(ComplexRMSELoss, self).__init__()

    def forward(self, input, target):
        if not (target.size() == input.size()):
            warnings.warn(
                f"Using a target size ({target.size()}) that is different to the input size ({input.size()}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.",
                stacklevel=2,
            )
        error_amplitude = (input - target).abs().float()
        zeros = torch.zeros_like(error_amplitude).float()
        mse = F.mse_loss(error_amplitude, zeros)
        rmse = torch.sqrt(mse)
        return rmse