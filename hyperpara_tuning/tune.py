"""Hyperparameter tuning using Optuna for Age-Gender prediction model."""

import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json

from model.model import AgeGenderModel
from model.losses import MultiTaskLoss
from preprocess.dataloader import build_loaders
from trainer.trainer import Trainer


class HyperparameterTuner:
    """Tune hyperparameters using Optuna."""
    
    def __init__(
        self,
        data_dir: str = "./labeled",
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        n_trials: int = 20,
        n_epochs_per_trial: int = 10,
    ):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            data_dir: Path to labeled data directory.
            batch_size: Batch size for training.
            num_workers: Number of workers for data loading.
            device: Device to use (cpu/cuda).
            checkpoint_dir: Directory to save checkpoints.
            n_trials: Number of trials to run.
            n_epochs_per_trial: Number of epochs per trial.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir
        self.n_trials = n_trials
        self.n_epochs_per_trial = n_epochs_per_trial
        
        # Load data once
        self.train_loader, self.val_loader, self.test_loader = build_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        
        # Create tuning directory
        self.tuning_dir = Path(checkpoint_dir) / "tuning_results"
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_params = None
        self.best_val_loss = float("inf")
    
    def objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna.
        
        Args:
            trial: Optuna trial object.
        
        Returns:
            Validation loss (to be minimized).
        """
        # Suggest hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        alpha = trial.suggest_float("alpha", 0.1, 2.0)
        beta = trial.suggest_float("beta", 0.1, 2.0)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        
        # Create model, loss, optimizer
        model = AgeGenderModel(pretrained=True)
        criterion = MultiTaskLoss(alpha=alpha, beta=beta)
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            checkpoint_dir=str(self.tuning_dir / f"trial_{trial.number}"),
            alpha=alpha,
            beta=beta,
        )
        
        # Train for n_epochs_per_trial
        for epoch in range(1, self.n_epochs_per_trial + 1):
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.evaluate()
            
            # Report intermediate value for pruning
            trial.report(val_metrics["loss"], step=epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # Return final validation loss
        val_metrics = trainer.evaluate()
        return val_metrics["loss"]
    
    def tune(self):
        """Run hyperparameter tuning."""
        print(f"Starting hyperparameter tuning with {self.n_trials} trials...")
        print(f"Each trial will run for {self.n_epochs_per_trial} epochs")
        print("-" * 80)
        
        # Create study
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best trial
        best_trial = study.best_trial
        self.best_params = best_trial.params
        self.best_val_loss = best_trial.value
        
        # Print results
        print("-" * 80)
        print(f"\nBest trial: Trial #{best_trial.number}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("\nBest hyperparameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        # Save results
        results = {
            "best_trial_number": best_trial.number,
            "best_validation_loss": self.best_val_loss,
            "best_hyperparameters": self.best_params,
            "n_trials": self.n_trials,
            "n_epochs_per_trial": self.n_epochs_per_trial,
        }
        
        results_path = self.tuning_dir / "tuning_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nTuning results saved to: {results_path}")
        
        return self.best_params
    
    def get_best_params(self):
        """Get best hyperparameters found."""
        return self.best_params


def main():
    """Main tuning script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune hyperparameters for Age-Gender Prediction Model")
    parser.add_argument("--data_dir", type=str, default="./labeled", help="Path to labeled data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs per trial")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    tuner = HyperparameterTuner(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        n_trials=args.n_trials,
        n_epochs_per_trial=args.n_epochs,
    )
    
    best_params = tuner.tune()
    print("\nTuning completed successfully!")


if __name__ == "__main__":
    main()
