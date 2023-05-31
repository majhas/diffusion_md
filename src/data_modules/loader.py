import torch
import mdtraj as mdt

from torch.utils.data import Dataset

class MinMaxTransform:
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value


    def standardize(self, x):
        return (x - self.min) / (self.max - self.min)

    def unstandardize(self, x):
        return x* (self.max - self.min) + self.min

    
class WhitenTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def standardize(self, x):
        return (x - self.mean) / self.std

    def unstandardize(self, x):
        return (x * self.std) + self.mean


class TorsionLoader(Dataset):
    def __init__(self, filepath=None, dataset=None, transform='whiten'):
        
        assert (filepath is None) ^ (dataset is None)

        if filepath is not None:
            dataset = np.load(filepath)

        self.dataset = torch.from_numpy(dataset).float()

        if transform == 'minmax':
            min_value = self.dataset.min(0).item()
            max_value = self.dataset.max(0).item()
            self.transform = MinMaxTransform(min_value, max_value)

        elif transform == 'whiten':
            mean = self.dataset.mean(0).item()
            std = self.dataset.std(0).item()
            self.transform = WhitenTransform(mean, std)
        
        self.standardized_dataset = self.transform(self.dataset)

        if len(self.standardized_dataset.shape) < 2:
            self.standardized_dataset = self.standardized_dataset.unsqueeze(1)

    def __getitem__(self, index):
        return self.standardized_dataset[index]

    def __len__(self):
        return self.standardized_dataset.size(0)



class SimulationLoader(Dataset):
    def __init__(self, filepath=None, dataset=None, transform=None):
        super(SimulationLoader, self).__init__()
        
        assert (filepath is None) ^ (dataset is None)

        if filepath is not None:
            dataset = np.load(filepath)

        self.dataset = torch.from_numpy(dataset).float()
        self.transform = None

        if transform == 'minmax':
            min_value = self.dataset.min(0).item()
            max_value = self.dataset.max(0).item()
            self.transform = MinMaxTransform(min_value, max_value)

        elif transform == 'whiten':
            mean = self.dataset.mean(0).item()
            std = self.dataset.std(0).item()
            self.transform = WhitenTransform(mean, std)

        if self.transform is not None:
            self.dataset = self.transform(self.dataset)

        if len(self.dataset.shape) < 2:
            self.dataset = self.dataset.unsqueeze(1)

        n_atoms = self.dataset.size(-2)
        n_dim = self.dataset.size(-1)

        src = self.dataset[:-1].view(-1, 1, n_atoms, n_dim)
        dst = self.dataset[1:].view(-1, 1, n_atoms, n_dim)
        self.dataset_pairs = torch.cat([src, dst], dim=1)


    def __getitem__(self, index):
        return self.dataset_pairs[index]


    def __len__(self):
        return self.dataset.size(0)


