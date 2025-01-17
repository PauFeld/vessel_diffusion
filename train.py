import argparse
import os

from pathlib import Path

import torch
from torch.backends import cudnn

from tqdm import tqdm

from utils.checkpoint import save_checkpoint
from datasets.vessel_set import VesselSet, AneuriskVesselSet, IntraVesselSet
from modules.edm import EDMLoss, EDMPrecond

import wandb


def parse_arguments(data_path):
    parser = argparse.ArgumentParser(prog="train vessel diffusion")

    # training params
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--warmup_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)

    # model params
    parser.add_argument("--num_points", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_channels", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--model_id", type=str, default="")

    # misc
    parser.add_argument("--data_path", type=str, default=data_path)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
    parser.add_argument("--val_iter", type=int, default=10)
    parser.add_argument("--save_checkpoint_iter", type=int, default=500)
    parser.add_argument(
        "--logging_mode",
        type=str,
        default="online",
        choices=["disabled", "online", "offline"],
    )
    parser.add_argument("--logging_id", type=str, default="vessel_diffusion")

    return parser.parse_args()

def calculate_parameter_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_memory = total_params * 4  # Assuming float32 (4 bytes)
    print("total number of parameters:", total_params)
    return total_memory

# Calculate memory usage during a forward pass for a single batch
def calculate_activation_memory(model, inputs, criterion, labels):
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        loss, _ = criterion(model, inputs, labels=labels)
    peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return peak_memory

# Estimate optimizer state memory (for Adam optimizer)
def calculate_optimizer_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    optimizer_memory = total_params * 4 * 3  # Adam uses 3x parameter size (float32)
    return optimizer_memory

# Accumulate activation memory over an epoch


def train_epoch(model, optimizer, criterion, scheduler, train_loader, args):
    model.train()

    pbar = tqdm(train_loader)
    '''
    for i, (inputs, labels) in enumerate(train_loader):
        print(f'Batch {i} input shapes:', [input.shape for input in inputs])
        if any(input.shape != torch.Size([128, 8]) for input in inputs):
            print(f'Inconsistent shapes found in batch {i}')'''

    for inputs, labels in pbar:
        optimizer.zero_grad()

        inputs = inputs.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        loss, _ = criterion(model, inputs, labels=labels)
        param_memory = calculate_parameter_memory(model)
        optimizer_memory = calculate_optimizer_memory(model)
        epoch_activation_memory = calculate_activation_memory(model, inputs, criterion, labels)

        total_memory = param_memory + optimizer_memory + epoch_activation_memory

        print(f"Parameter Memory: {param_memory / (1024 ** 2):.2f} MB")
        print(f"Optimizer Memory: {optimizer_memory / (1024 ** 2):.2f} MB")
        print(f"Activation Memory Over Epoch: {epoch_activation_memory / (1024 ** 2):.2f} MB")
        print(f"Total Estimated Memory Over Epoch: {total_memory / (1024 ** 2):.2f} MB")

        loss.backward()
        optimizer.step()

        wandb.log({"train": {"loss": loss.item(), "lr": scheduler.get_last_lr()[0]}})
        pbar.set_description(f"Training - loss: {loss.item():.3f}")

        scheduler.step()


@torch.no_grad()
def val_epoch(model, criterion, val_loader, args):
    model.eval()

    total_loss = 0
    
    for i, (inputs, labels) in enumerate(tqdm(val_loader, desc="Validating"), start=1):
        inputs = inputs.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)

        loss, _ = criterion(model, inputs, labels=labels)

        total_loss += loss.item()

    metrics = {"val": {"loss": total_loss / i}}

    wandb.log({"val": {"loss": loss}}, commit=False)

    return metrics


def train(model, optimizer, criterion, scheduler, train_loader, val_loader, args):
    val_epoch(model, criterion, val_loader, args)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch: {epoch}/{args.epochs}")

        train_epoch(model, optimizer, criterion, scheduler, train_loader, args)

        if epoch % args.val_iter == 0:
            metrics = val_epoch(model, criterion, val_loader, args)

        if epoch % args.save_checkpoint_iter == 0 or epoch == args.epochs:
            save_checkpoint(
                model,
                epoch,
                os.path.join(args.checkpoint_path),
                metrics=metrics,
                model_kwargs=args.model_kwargs,
            )


def main():
    model_name = "intra"
    args = parse_arguments(model_name)

    #
    # Training environment setup
    #
    assert args.warmup_epochs <= args.epochs

    default_device = "cuda" if torch.cuda.is_available() else None
    args.device = args.device if args.device else default_device

    cudnn.benchmark = True

    #model_name = (
    #    f"edm_n{args.num_points}_c{args.num_channels}_d{args.depth}_h{args.num_heads}_cl{args.num_classes}"
    #    + (f"_{args.model_id}" if args.model_id else "")
    #)

    print("model name", model_name)
    checkpoint_path = os.path.join(args.checkpoint_path, model_name)

    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    args.checkpoint_path = checkpoint_path
    print("checkpoint path", args.checkpoint_path)
    wandb.init(
        project=args.logging_id, entity="paufeldman", name=model_name, config=args, mode=args.logging_mode
    )

    #
    # Data setup
    #
    if model_name == "aneurisk":
        train_set = AneuriskVesselSet(split="train", path=args.data_path)
        val_set = AneuriskVesselSet(split="test", path=args.data_path)
        print(f"path: {train_set.path}")
    elif model_name == "intra":
        train_set = IntraVesselSet(split="train", path=args.data_path)
        val_set = IntraVesselSet(split="test", path=args.data_path)
    elif model_name == "dummy_data":
        train_set = VesselSet(split="train", path=args.data_path)
        val_set = VesselSet(split="test", path=args.data_path)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    #
    # Model setup
    #
    model_kwargs = {
        "n_points": args.num_points,
        "channels": args.num_channels,
        "n_heads": args.num_heads,
        "d_head": args.dim_head,
        "depth": args.depth,
        "num_classes": args.num_classes,
    }
    args.model_kwargs = model_kwargs

    model = EDMPrecond(**model_kwargs).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = EDMLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        pct_start=args.warmup_epochs / args.epochs,
        total_steps=args.epochs * len(train_loader),
        div_factor=args.lr / args.min_lr,
        final_div_factor=1,
    )

    train(model, optimizer, criterion, scheduler, train_loader, val_loader, args)


if __name__ == "__main__":
    main()
