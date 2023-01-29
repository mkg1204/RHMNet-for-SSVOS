import random
import torch.utils.data as data

class SeqSampler(data.Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, init_skip=0, temporal_flip=False):
        self.datasets = datasets
        self.skip = init_skip
        self.temporal_flip = temporal_flip
        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]
        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def change_skip(self, f):
        self.skip = f
        for dataset in self.datasets:
            dataset.change_skip(f)
    
    def flip(self, Fs, Gs, Init_Scr, Ms, Ds_Ms, Ignore_R):
        Fs = Fs.flip(dims=(0,))
        Gs = Gs.flip(dims=(0,))
        Init_Scr = Init_Scr.flip(dims=(0,))
        Ms = Ms.flip(dims=(0,))
        Ds_Ms = Ds_Ms.flip(dims=(0,))
        Ignore_R = Ignore_R.flip(dims=(0,))
        return Fs, Gs, Init_Scr, Ms, Ds_Ms, Ignore_R

    def __getitem__(self, index):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        seq_id = random.randint(0, len(dataset)-1)
        Fs, Gs, Init_Scr, Ms, Ds_Ms, Ignore_R, num_object, info = dataset.__getitem__(seq_id)
        if self.temporal_flip and random.uniform(0, 1) > 0.5:
            Fs, Gs, Init_Scr, Ms, Ds_Ms, Ignore_R = self.flip(Fs, Gs, Init_Scr, Ms, Ds_Ms, Ignore_R)
        return Fs, Gs, Init_Scr, Ms, Ds_Ms, Ignore_R, num_object, info
