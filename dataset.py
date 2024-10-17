from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


class XORDataset(Dataset):
    def __init__(self, num_samples):
        original_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        original_outputs = np.array([0, 1, 1, 0], dtype=np.float32).reshape(-1, 1)
        self.inputs = np.tile(original_inputs, (num_samples // len(original_inputs) + 1, 1))[:num_samples]
        self.outputs = np.tile(original_outputs, (num_samples // len(original_outputs) + 1, 1))[:num_samples]
        if len(self.inputs) > num_samples:
            self.inputs = self.inputs[:num_samples]
            self.outputs = self.outputs[:num_samples]
        indices = np.arange(len(self.inputs))
        np.random.shuffle(indices)
        self.inputs = self.inputs[indices]
        self.outputs = self.outputs[indices]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {'input': self.inputs[idx], 'output': self.outputs[idx]}
        return sample
