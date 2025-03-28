import torchvision
import torchvision.transforms as transforms

import datasets.omniglot as om


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, train=True, path=None, background=True, all=False):

        if name == "omniglot":
            train_transform = transforms.Compose(
                [transforms.Resize((84, 84)),
                 transforms.ToTensor()])
            if path is None:
                return om.Omniglot("../data/omni", background=background, download=True, train=train,
                                   transform=train_transform, all=all)
            else:
                return om.Omniglot(path, background=background, download=True, train=train,
                                   transform=train_transform, all=all)

        elif name == "svhn":
            if train:
                return torchvision.datasets.SVHN(root='../data/SVHN', split='train', download=True,
                                                 transform=transforms.ToTensor())
            else:
                return torchvision.datasets.SVHN(root='../data/SVHN', split='test', download=True,
                                                 transform=transforms.ToTensor())

        else:
            print("Unsupported Dataset")
            assert False
