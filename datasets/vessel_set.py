import torch
import numpy as np

import os


class VesselSet(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str = "train",
        num_points: int = 128,
        merge_m2_branches: bool = True,
        path: str = "dummy_data",
        data_scaler: float = 24,
        zero_mean_data: bool = True,
    ):
        assert split in ["train", "test"]

        self.num_points = num_points
        self.merge_m2_branches = merge_m2_branches
        self.data_scaler = data_scaler
        self.zero_mean_data = zero_mean_data

        #self.path = os.path.join(path, f"data_{split}.npy")
        self.path = "datasets/" + path + "/data_" + split + ".npy"
        self.split = split

        self.data = np.load(self.path)

    def __getitem__(self, index):
        tree = self.data[index]

        label = int(index % 2 == 0)

        # discard m2 vessels if label == 0 (occlusion is present)
        tree = tree if label == 1 else tree[:-2]

        # calculate number of points per vessel segment
        points_per_segment = [segment.shape[0] for segment in tree]

        total_points = sum(points_per_segment)

        sample_points_per_segment = [
            int((n / total_points) * (self.num_points)) for n in points_per_segment
        ]

        sample_points_per_segment[
            np.argmax(sample_points_per_segment)
        ] += self.num_points - sum(sample_points_per_segment)

        uniform_tree = []

        # interpolate points for equidistant sampling
        for segment, num_sample_points in zip(tree, sample_points_per_segment):
            segment_rows = []

            dists = np.array(
                [0, *np.linalg.norm(segment[:-1, :3] - segment[1:, :3], axis=-1)]
            )
            s = np.cumsum(dists)

            points = np.linspace(0, s[-1], num_sample_points)

            for row in segment.T:
                segment_rows.append(np.interp(points, s, row))

            uniform_tree.append(np.vstack(segment_rows).T)

        # get one-hot labels
        typed_tree = []
        num_types = 4 if self.merge_m2_branches else 5
        for vessel_type, segment in enumerate(uniform_tree):
            if vessel_type == 4 and self.merge_m2_branches:
                vessel_type = 3

            one_hot = (
                np.zeros((segment.shape[0], 1)) + vessel_type - np.arange(num_types)
            )
            one_hot = (one_hot == 0).astype(float)

            typed_tree.append(np.concatenate((segment, one_hot), axis=-1))

        tree = torch.from_numpy(np.concatenate(typed_tree, axis=0))

        # normalize data
        if self.zero_mean_data:
            tree[:, :3] -= tree[:, :3].mean(0, keepdims=True)

        tree[:, :4] /= self.data_scaler
        return tree, label

    def __len__(self):
        return len(self.data)

class AneuriskVesselSet(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str = "train",
        num_points: int = 128,
        merge_m2_branches: bool = True,
        path: str = "dummy_data",
        data_scaler: float = 24,
        zero_mean_data: bool = True,
    ):
        assert split in ["train", "test"]

        self.num_points = num_points
        self.merge_m2_branches = merge_m2_branches
        self.data_scaler = data_scaler
        self.zero_mean_data = zero_mean_data

        #self.path = os.path.join(path, f"data_{split}.npy")
        self.path = "datasets/" + path + "/data_" + split + ".npy"
        self.split = split

        self.data = np.load(self.path)

    def __getitem__(self, index):
        tree = self.data[index]
        
        label = int(index % 2 == 0)
        
        # Discard M2 vessels if label == 0 (occlusion is present)
        tree = tree if label == 1 else tree[:-2]

        # Filter out completely padded segments
        valid_segments = []
        for segment in tree:
            # Remove rows with all zeros (padded points)
            valid_points = segment[np.any(segment != 0, axis=-1)]
            if valid_points.size > 0:  # Include segment only if it has valid points
                valid_segments.append(valid_points)

        # Calculate number of points per valid segment
        points_per_segment = [len(segment) for segment in valid_segments]
        total_points = sum(points_per_segment)
        
        # Sample points per segment based on the required total number of points
        sample_points_per_segment = [
            int((n / total_points) * self.num_points) for n in points_per_segment
        ]
        sample_points_per_segment[np.argmax(sample_points_per_segment)] += (
            self.num_points - sum(sample_points_per_segment)
        )
        # Interpolate points for equidistant sampling
        uniform_tree = []
        masks = []  # Store masks for valid points

        for segment, num_sample_points in zip(valid_segments, sample_points_per_segment):
            if num_sample_points > len(segment):
                # If requested sample points exceed segment points, adjust as necessary
                num_sample_points = len(segment)

            # Calculate distances and cumulative distances
            dists = np.concatenate(([0], np.linalg.norm(segment[:-1, :3] - segment[1:, :3], axis=-1)))
            s = np.cumsum(dists)

            # Interpolated points
            points = np.linspace(0, s[-1], num_sample_points)
            sampled_segment = np.array([
                np.interp(points, s, segment[:, i]) for i in range(segment.shape[1])
            ]).T

            # Append to uniform tree and generate mask
            uniform_tree.append(sampled_segment)
            mask = np.ones((num_sample_points, 1))  # Mark as valid points
            masks.append(mask)

        
        # Process typed_tree with one-hot encoding
        typed_tree = []
        num_types = 4 #if self.merge_m2_branches else 5
        for vessel_type, segment in enumerate(uniform_tree):
            # Adjust for merged M2 branches
            #if vessel_type == 4 and self.merge_m2_branches:
            #    vessel_type = 3
            # Generate one-hot encoding for the vessel type
            one_hot = np.zeros((segment.shape[0], num_types))
            one_hot[:, vessel_type] = 1

            # Append one-hot encoded vessel type to the segment data
            typed_tree.append(np.concatenate((segment, one_hot), axis=-1))
        
        # Concatenate segments into a single array and convert to tensor
        tree = torch.from_numpy(np.concatenate(typed_tree, axis=0))

        # Normalize data
        if self.zero_mean_data:
            tree[:, :3] -= tree[:, :3].mean(0, keepdims=True)

        tree[:, :4] /= self.data_scaler

        # Concatenate masks into a single array and convert to tensor
        masks = torch.from_numpy(np.concatenate(masks, axis=0)).float()

        # Return tree, label, and masks
        return tree, label#, masks

    def __len__(self):
        return len(self.data)


from matplotlib import pyplot as plt

def main():
    

    dataset = VesselSet()
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    vessel, label = next(iter(loader))
    #print(vessel.shape, label)

    points, radii, types = vessel[0, :, :3], vessel[0, :, 3], vessel[0, :, 4:]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*points.T, s=(220 * radii) ** 2, c=types.argmax(-1), vmin=0, vmax=3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.show()


if __name__ == "__main__":
    main()
    ves = np.load("datasets/dummy_data/data_test.npy")
    #print(ves.shape)
    #ves.shape = (40, 5, 256, 4)
    #cada arbol tiene tamaño (5, 256, 4) 5 ramas de 256 puntos cada una y cuatro atributos (x, y, z, r)
