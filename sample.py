import argparse
import os

import numpy as np

import torch

from tqdm import trange

from matplotlib import pyplot as plt

from modules.edm import EDMPrecond
from utils.checkpoint import load_checkpoint
from centerline_sequencer.centerline_sequencer import merge_centerline
import time

def parse_arguments(dataset):
    parser = argparse.ArgumentParser(prog="train vessel diffusion")

    parser.add_argument("--model_name", type=str, default=dataset)

    parser.add_argument("--sample_class", type=int, default=0)

    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--checkpoint_epoch", type=int, default=2500)
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
        ''''
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

        plt.show()'''
        return out


def main(dataset):
    args = parse_arguments(dataset)
    args.model_name = dataset
    default_device = "cuda" if torch.cuda.is_available() else None
    args.device = args.device if args.device else default_device

    args.num_samples = args.num_samples if args.seed < 0 else 1

    checkpoint_path = os.path.join(
        args.checkpoint_path, dataset, f"checkpoint-{args.checkpoint_epoch}.pth"
    )
    print(args.model_name)
    #checkpoint_path = args.checkpoint_path +"/" + args.model_name + "/checkpoint-2500.pth"
    #checkpoint_path = "checkpoints/intra/checkpoint-2500.pth"

    model = load_checkpoint(EDMPrecond, checkpoint_path, device=args.device)
    model.eval()
    n_samples = 10
    tensors_list = []
    
    for i in range(n_samples):
        try:
            start_time = time.time()
            out= sample(model, args)
            end_time = time.time()

            # Calculate and print the elapsed time
            elapsed_time = end_time - start_time
            print(f"Execution time: {elapsed_time:.2f} seconds")
            tensors_list.append(out)
        except:
            print("failed")
        
        

    generated_set = torch.stack(tensors_list, dim=0).numpy()
   
    print(generated_set.shape)
    np.save("generados/" + dataset + "/set1.npy", generated_set)

def save_segments_and_radii_to_text(segments, radii, segment_filename, radii_filename):
    # Save segments to a text file
    with open(segment_filename, 'w') as seg_file:
        for segment in segments:
            np.savetxt(seg_file, segment, delimiter=',')
            seg_file.write("# New Segment\n")  # Mark the end of a segment
    
    # Save radii to a text file
    with open(radii_filename, 'w') as rad_file:
        for radius_array in radii:
            np.savetxt(rad_file, radius_array.reshape(1, -1), delimiter=',')
            rad_file.write("# New Radii\n")  # Mark the end of a radii array

if __name__ == "__main__":
    
    dataset = "intra"
    main(dataset)
    generadas = np.load("generados/" + dataset + "/set1.npy")
    output_dir = "generados/" + dataset + "/plot_generadas_" + dataset
    for j, out in enumerate(generadas):
       
        coords, radii, vessel_types = map(
            lambda x: x, (out[..., :3], out[..., 3], out[..., :4])
        )
        vessel_types = np.argmax(vessel_types, axis=-1)
        try:
            centerline_segments, centerline_features = merge_centerline(
                coords, radii[..., None], vessel_types
            )
            print(centerline_segments)
            print("radius", centerline_features)
            
            o_dir = "generados/" + dataset 
            save_segments_and_radii_to_text(centerline_segments, centerline_features, o_dir +f"/segments/segments_{j}.txt", o_dir +f"/features/radii_{j}.txt")


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


        except:
            print("failed merging")

               
        # Save the plot with a unique filename
        filename = os.path.join(output_dir, f"centerline_plot_{j}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved plot {j} to {filename}")

        plt.show()
        #if j > 1:
        #    break