import os
import torch.utils.data as data
from PIL import Image
import torch

class ContourDataset(data.Dataset):
    def __init__(self,
                 img_path,
                 skt_path,
                 list_path,
                 transformer=None):
        self.img_path = img_path
        self.skt_path = skt_path

        with open(list_path) as f:
            content = f.readlines()
        self.filelist = sorted([x.strip() for x in content])
        self.transformer = transformer
        self.N = 5

    def __getitem__(self, index):
        filename = self.filelist[index]

        pathA = os.path.join(self.img_path, filename + '.jpg')
        image = Image.open(pathA)

        targets = []

        for i in range(self.N):
            pathB = os.path.join(self.skt_path, '%s_%02d.png' % (filename, i+1))
            target = Image.open(pathB)

            targets.append(target)

        if self.transformer is not None:
            image, targets = self.transformer(image, targets)

        targets = torch.cat(targets, 0)

        return image, targets

    def __len__(self):
        return len(self.filelist)
