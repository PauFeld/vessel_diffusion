import argparse
import os

import numpy as np

import torch

from tqdm import trange

from matplotlib import pyplot as plt

from modules.edm import EDMPrecond
from utils.checkpoint import load_checkpoint
from centerline_sequencer.centerline_sequencer import merge_centerline


def parse_arguments():
    parser = argparse.ArgumentParser(prog="train vessel diffusion")

    parser.add_argument("--model_name", type=str, default="edm_n128_c8_d6_h8_cl2")

    parser.add_argument("--sample_class", type=int, default=0)

    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--checkpoint_epoch", type=int, default=10)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints")

    parser.add_argument("--device", type=str, default="")

    return parser.parse_args()


@torch.no_grad()
def sample(model, args):
    for _ in trange(args.num_samples):
        seed = (
            torch.randint(0, 99999, (1,), device=args.device)
            if args.seed < 0
            else torch.as_tensor(args.seed, device=args.device)
        )
        label = torch.tensor([args.sample_class], device=args.device)

        print(
            f"\nSampling centerline with seed {seed.item()} and label {label.item()}."
        )

        out = model.sample(cond=label, batch_seeds=seed)[0].cpu()

        #
        # Preparing model output for centerline sequencing
        #
        coords, radii, vessel_types = map(
            lambda x: x.numpy(), (out[..., :3], out[..., 3], out[..., :4])
        )
        vessel_types = np.argmax(vessel_types, axis=-1)

        centerline_segments, centerline_features = merge_centerline(
            coords, radii[..., None], vessel_types
        )

        #
        # Visualizing centerline sequences
        #
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        for i, (segment, radii) in enumerate(
            zip(centerline_segments, centerline_features)
        ):
            ax.plot(*segment.T, label=i)
            ax.scatter(*segment.T, s=500 * np.abs(radii), alpha=0.3)

        plt.legend()
        plt.tight_layout()
        plt.axis("off")
        plt.savefig('centerline_plot.png', dpi=300, bbox_inches='tight')  # Save as PNG with 300 DPI

        plt.show()
        


def main():
    args = parse_arguments()

    default_device = "cuda" if torch.cuda.is_available() else None
    args.device = args.device if args.device else default_device

    args.num_samples = args.num_samples if args.seed < 0 else 1

    #checkpoint_path = os.path.join(
    #    args.checkpoint_path, args.model_name, f"checkpoint-{args.checkpoint_epoch}.pth"
    #)
    checkpoint_path = args.checkpoint_path +"/" + args.model_name + "/checkpoint-500.pth"

    model = load_checkpoint(EDMPrecond, checkpoint_path, device=args.device)
    model.eval()

    sample(model, args)


if __name__ == "__main__":
    main()
