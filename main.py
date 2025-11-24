"""Main script to train Age- Gender prediction model."""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Import custom modules
from model.model import AgeGenderModel
from model.losses import MultiTaskLoss
from preprocess.dataloader import build_loaders
from trainer.trainer import Trainer


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train Age- Gender Prediction Model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./labeled", help="Path to labeled data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    
    # Model arguments
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained ResNet50")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # Loss weight arguments
    parser.add_argument("--alpha", type=float, default=2.0, help="Weight for age loss")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for gender loss")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use") # if torch.cuda.is_available() else "cpu"
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_dir}...")
    train_loader, val_loader, test_loader = build_loaders(
        train_csv=f"{args.data_dir}/train/label.csv",
        train_img=f"{args.data_dir}/train/img",
        val_csv=f"{args.data_dir}/valid/label.csv",
        val_img=f"{args.data_dir}/valid/img",
        test_csv=f"{args.data_dir}/test/label.csv",
        test_img=f"{args.data_dir}/test/img",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=True,
    )
    print(f"- Data loaded successfully")
    
    # Create model
    print("Creating model...")
    model = AgeGenderModel(pretrained=args.pretrained)
    model = model.to(device)
    print(f"- Model created: AgeGenderModel (ResNet50)")
    
    # Create loss function
    criterion = MultiTaskLoss(alpha=args.alpha, beta=args.beta)
    print(f"- Loss function: MultiTaskLoss (alpha={args.alpha}, beta={args.beta})")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print(f"- Optimizer: Adam (lr={args.learning_rate}, weight_decay={args.weight_decay})")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        alpha=args.alpha,
        beta=args.beta,
    )
    
    # Train model
    print("\n" + "="*80)
    trainer.fit(epochs=args.epochs, patience=args.patience)
    print("="*80 + "\n")
    
    print("Training completed successfully!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
