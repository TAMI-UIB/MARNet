from os import environ
from os.path import join
import h5py
import torch
import torch.utils.data as data

from torch.utils.data import DataLoader
import pytorch_lightning as pl



class WorldView2(data.Dataset):
    def __init__(self, subset, sampling):
        super(WorldView2, self).__init__()
        if sampling != 4:
            exit("Sampling must be 4")
        self.dataset_path = join(environ['DATASET_PATH'], "WorldView2")
        print(f'######### {self.dataset_path}/data.h5 #########')
        data = h5py.File(f'{self.dataset_path}/data.h5')


        self.hs = data[subset]['ms']
        self.pan =data[subset]['pan']
        self.subset = subset
        if subset != "fullres":
            self.gt = data[subset]['gt']

    def __getitem__(self, index):
        if self.subset != "fullres":
            gt = torch.tensor(self.gt[index, :, :, :], dtype=torch.float32)
        pan =  torch.tensor(self.pan[index, :, :, :], dtype=torch.float32)
        hs = torch.tensor(self.hs[index, :, :, :], dtype=torch.float32)
        if self.subset != "fullres":
            return {"pan": pan, "hs": hs}, {"gt": gt}, f"{self.subset}_{index}"
        else:
            return {"pan": pan, "hs": hs}, {}, f"{self.subset}_{index}"

    def __len__(self):
        return self.hs.shape[0]

    def get_rgb(self, tensor):
        C = tensor.size(0)
        if C > 1:
            tensor = tensor[[4, 2, 1], :, :]
            wb_input = self.white_balance_correction(tensor)
            gamma_red = 1.1
            gamma_green = 1.1
            gamma_blue = 1.1
            wb_input[0, :, :] = self.gamma_correction(wb_input[0, :, :], gamma_red)
            wb_input[1, :, :] = self.gamma_correction(wb_input[1, :, :], gamma_green)
            wb_input[2, :, :] = self.gamma_correction(wb_input[2, :, :], gamma_blue)
        elif C ==1:
            wb_input = torch.cat([tensor,tensor, tensor], dim=0)
        return wb_input

    def gamma_correction(self, input, gamma):
        return torch.pow(input, 1. / gamma)

    def white_balance_correction(self, input):
        r = input[[0], :, :]
        g = input[[1], :, :]
        b = input[[2], :, :]

        avg_r = torch.mean(r)
        avg_g = torch.mean(g)
        avg_b = torch.mean(b)

        scale_r = avg_g / avg_r
        scale_b = avg_g / avg_b

        corrected_r = torch.clamp(r * scale_r, 0, 1)
        corrected_g = g
        corrected_b = torch.clamp(b * scale_b, 0, 1)

        corrected_image = torch.cat([corrected_r, corrected_g, corrected_b], dim=0)
        return corrected_image

class WorldViewDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, sampling=4):
        super().__init__()
        self.sampling = sampling
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.fullres_dataset = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = WorldView2(sampling=self.sampling, subset='train')
            self.val_dataset = WorldView2(sampling=self.sampling, subset='validation')
            self.test_dataset = WorldView2(sampling=self.sampling, subset='test')
        if stage == "test":
            self.val_dataset = WorldView2(sampling=self.sampling, subset='validation')
            self.test_dataset = WorldView2(sampling=self.sampling, subset='test')
        if stage == "full_res":
            self.fullres_dataset = WorldView2(sampling=self.sampling, subset='fullres')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),

    def test_dataloader(self):
        if self.fullres_dataset is not None:
            dataloaders = [
                DataLoader(self.fullres_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            ]
        else:
            dataloaders = [
                DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
                DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                    ]
        return dataloaders
