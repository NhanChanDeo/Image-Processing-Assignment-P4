"""Trainer class for Age-Gender prediction model."""

import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import json
from datetime import datetime

from .metrics import get_gender_accuracy, get_age_mae, get_age_rmse


class Trainer:
    """
    Trainer class to manage the training and validation loop.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        """
        Initialize the Trainer.
        
        Args:
            model: PyTorch model to train.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            criterion: Loss function (MultiTaskLoss).
            optimizer: Optimizer.
            device: Device to use (cpu/cuda).
            checkpoint_dir: Directory to save checkpoints.
            alpha: Weight for age loss.
            beta: Weight for gender loss.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.alpha = alpha
        self.beta = beta
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Tracking metrics
        self.train_history = {
            "epoch": [],
            "train_loss": [],
            "train_gender_acc": [],
            "train_age_mae": [],
            "val_loss": [],
            "val_gender_acc": [],
            "val_age_mae": [],
        }
        
        self.best_val_loss = float("inf")
        self.best_epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.
        
        Returns:
            Dictionary with training metrics for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        all_gender_preds = []
        all_gender_labels = []
        all_age_preds = []
        all_age_labels = []
        
        for batch_idx, (images, ages, genders) in enumerate(self.train_loader):
            images = images.to(self.device)
            ages = ages.to(self.device).unsqueeze(1)  # shape: (batch_size, 1)
            genders = genders.to(self.device).float().unsqueeze(1)  # shape: (batch_size, 1)
            
            # Forward pass
            self.optimizer.zero_grad()
            age_pred, gender_pred = self.model(images)
            
            # Compute loss
            loss = self.criterion(age_pred, gender_pred, ages, genders)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_gender_preds.append(gender_pred.detach().cpu())
            all_gender_labels.append(genders.detach().cpu())
            all_age_preds.append(age_pred.detach().cpu())
            all_age_labels.append(ages.detach().cpu())
        
        # Compute epoch metrics
        all_gender_preds = torch.cat(all_gender_preds, dim=0)
        all_gender_labels = torch.cat(all_gender_labels, dim=0)
        all_age_preds = torch.cat(all_age_preds, dim=0)
        all_age_labels = torch.cat(all_age_labels, dim=0)
        
        avg_loss = total_loss / len(self.train_loader)
        gender_acc = get_gender_accuracy(all_gender_preds.squeeze(), all_gender_labels.squeeze())
        age_mae = get_age_mae(all_age_preds.squeeze(), all_age_labels.squeeze())
        
        return {
            "loss": avg_loss,
            "gender_acc": gender_acc,
            "age_mae": age_mae,
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run validation loop.
        
        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_gender_preds = []
        all_gender_labels = []
        all_age_preds = []
        all_age_labels = []
        
        with torch.no_grad():
            for images, ages, genders in self.val_loader:
                images = images.to(self.device)
                ages = ages.to(self.device).unsqueeze(1)
                genders = genders.to(self.device).float().unsqueeze(1)
                
                # Forward pass
                age_pred, gender_pred = self.model(images)
                
                # Compute loss
                loss = self.criterion(age_pred, gender_pred, ages, genders)
                
                # Track metrics
                total_loss += loss.item()
                all_gender_preds.append(gender_pred.detach().cpu())
                all_gender_labels.append(genders.detach().cpu())
                all_age_preds.append(age_pred.detach().cpu())
                all_age_labels.append(ages.detach().cpu())
        
        # Compute epoch metrics
        all_gender_preds = torch.cat(all_gender_preds, dim=0)
        all_gender_labels = torch.cat(all_gender_labels, dim=0)
        all_age_preds = torch.cat(all_age_preds, dim=0)
        all_age_labels = torch.cat(all_age_labels, dim=0)
        
        avg_loss = total_loss / len(self.val_loader)
        gender_acc = get_gender_accuracy(all_gender_preds.squeeze(), all_gender_labels.squeeze())
        age_mae = get_age_mae(all_age_preds.squeeze(), all_age_labels.squeeze())
        
        return {
            "loss": avg_loss,
            "gender_acc": gender_acc,
            "age_mae": age_mae,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best checkpoint.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, best_path)
            print(f"✓ Saved best model at epoch {epoch}")
    
    def fit(self, epochs: int, patience: int = 10):
        """
        Fit the model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train.
            patience: Early stopping patience (number of epochs without improvement).
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training set size: {len(self.train_loader.dataset)}")
        print(f"Validation set size: {len(self.val_loader.dataset)}")
        print("-" * 80)
        
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.evaluate()
            
            # Update history
            self.train_history["epoch"].append(epoch)
            self.train_history["train_loss"].append(train_metrics["loss"])
            self.train_history["train_gender_acc"].append(train_metrics["gender_acc"])
            self.train_history["train_age_mae"].append(train_metrics["age_mae"])
            self.train_history["val_loss"].append(val_metrics["loss"])
            self.train_history["val_gender_acc"].append(val_metrics["gender_acc"])
            self.train_history["val_age_mae"].append(val_metrics["age_mae"])
            
            # Print metrics
            print(f"Epoch [{epoch}/{epochs}]")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | "
                  f"Gender Acc: {train_metrics['gender_acc']:.4f} | "
                  f"Age MAE: {train_metrics['age_mae']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | "
                  f"Gender Acc: {val_metrics['gender_acc']:.4f} | "
                  f"Age MAE: {val_metrics['age_mae']:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping check
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ Best validation loss improved!")
            else:
                patience_counter += 1
                print(f"  ⚠ No improvement for {patience_counter}/{patience} epochs")
            
            print("-" * 80)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
        
        print(f"\nTraining completed!")
        print(f"Best model at epoch {self.best_epoch} with validation loss: {self.best_val_loss:.4f}")
        
        # Save training history
        history_path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.train_history, f, indent=4)
        print(f"Training history saved to {history_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        print(f"Loaded checkpoint from epoch {epoch}")
        return epoch
