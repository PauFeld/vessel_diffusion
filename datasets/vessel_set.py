import torch


class VesselSet(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        self.split = split

    def __getitem__(self, index):
        return 0.5 * torch.randn(256, 8), torch.randint(0, 2, (1,), dtype=int).item()

    def __len__(self):
        return 100
