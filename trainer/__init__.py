"""Trainer module for Age-Gender prediction model."""

from .trainer import Trainer
from .metrics import get_gender_accuracy, get_age_mae

__all__ = ["Trainer", "get_gender_accuracy", "get_age_mae"]
