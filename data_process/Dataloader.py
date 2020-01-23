from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image


class myDataset(Dataset):
    def __init__(self, datapath='datapath', type='train'):
        self.datapath = datapath
        self.type = type

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img, label1, label2 = self.make_data(index)
        sample = {'image': img, 'label1': label1, 'label2': label2}
        return sample

    def make_data(self, index):
        img = Image.open(self.datapath[index])
        label1 = Image.open(self.datapath[index])
        label2 = Image.open(self.datapath[index])
        return img, label1, label2


def make_data_loader(batch_size, drop_last):
    train_set = myDataset('datapath', 'train')
    val_set = myDataset('datapath', 'val')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=drop_last)
    return train_loader, val_loader


