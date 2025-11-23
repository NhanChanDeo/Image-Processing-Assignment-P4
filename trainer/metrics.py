"""Evaluation metrics for Age-Gender prediction."""

import torch
import torch.nn.functional as F


def get_gender_accuracy(pred: torch.Tensor, label: torch.Tensor) -> float:
    """
    Calculate gender classification accuracy.
    
    Args:
        pred: Model output for gender (logits or probabilities), shape (batch_size,)
        label: Ground truth gender labels (0 or 1), shape (batch_size,)
    
    Returns:
        Accuracy as a float value between 0 and 1.
    """
    # Convert logits to binary predictions (threshold = 0.5)
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    correct = (pred_binary == label).float()
    accuracy = correct.mean().item()
    return accuracy


def get_age_mae(pred: torch.Tensor, label: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error for age prediction.
    
    Args:
        pred: Model predictions for age, shape (batch_size,)
        label: Ground truth age labels, shape (batch_size,)
    
    Returns:
        MAE as a float value.
    """
    mae = torch.abs(pred - label).mean().item()
    return mae


def get_age_rmse(pred: torch.Tensor, label: torch.Tensor) -> float:
    """
    Calculate Root Mean Squared Error for age prediction.
    
    Args:
        pred: Model predictions for age, shape (batch_size,)
        label: Ground truth age labels, shape (batch_size,)
    
    Returns:
        RMSE as a float value.
    """
    mse = torch.mean((pred - label) ** 2).item()
    rmse = mse ** 0.5
    return rmse
