from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms


class ButterflyMothDataset(Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root: the root directory of the dataset
            mode : [train, valid, test]
        Data transformation:
            Transform the .jpg rgb images during the training phase, such as resizing, random flipping,
            rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints.

            In the testing phase, if you have a normalization process during the training phase, you only need
            to normalize the data.

            hints:  Convert the pixel value to [0, 1]
                    Transpose the image shape from [H, W, C] to [C, H, W]
        """

        # define data transformation
        if mode == "train" or mode == "valid":
            self.transform = transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(p=0.3),
                    # transforms.RandomVerticalFlip(p=0.3),
                    # transforms.RandomRotation(degrees=20),
                    # transforms.ColorJitter(
                    #     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                    # ),
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )
        if mode == "test":
            self.transform = transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        # read data
        df = pd.read_csv(f"{root}/{mode}.csv")
        image_names = df["filepaths"].tolist()
        self.labels = df["label_id"].tolist()

        print(f"Loading {mode} dataset | Total {len(image_names)} images.")

        self.images = []
        for image_name in tqdm(image_names):
            image = Image.open(f"{root}/{image_name}")
            image = self.transform(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        return image, label


# class ButterflyMothDataset(Dataset):
#     def __init__(self, root, mode):
#         """
#         Args:
#             root: the root directory of the dataset
#             mode : [train, valid, test]
#         """
#         df = pd.read_csv(f"{root}/{mode}.csv")
#         self.image_names = df["filepaths"].tolist()
#         self.labels = df["label_id"].tolist()

#         self.image_names = [
#             f"{root}/{self.image_names[i]}" for i in range(len(self.image_names))
#         ]

#         # define data transformation
#         if mode == "train" or mode == "valid":
#             self.transform = transforms.Compose(
#                 [
#                     # transforms.RandomHorizontalFlip(p=0.3),
#                     # transforms.RandomRotation(degrees=20),
#                     # transforms.RandomVerticalFlip(),
#                     # transforms.RandomRotation(degrees=45),
#                     # transforms.ColorJitter(
#                     #     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
#                     # ),
#                     transforms.ToImage(),
#                     transforms.ToDtype(torch.float32, scale=True),
#                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                 ]
#             )
#         if mode == "test":
#             self.transform = transforms.Compose(
#                 [
#                     transforms.ToImage(),
#                     transforms.ToDtype(torch.float32, scale=True),
#                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                 ]
#             )

#         print(f"Loading {mode} dataset | Total {len(self.image_names)} images.")

#     def __len__(self):
#         return len(self.image_names)

#     def __getitem__(self, index):
#         """
#         step1. Get the image path from 'self.img_name' and load it.
#                hint : path = root + self.img_name[index] + '.jpg'

#         step2. Get the ground truth label from self.label

#         step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping,
#                rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints.

#                In the testing phase, if you have a normalization process during the training phase, you only need
#                to normalize the data.

#                hints : Convert the pixel value to [0, 1]
#                        Transpose the image shape from [H, W, C] to [C, H, W]

#          step4. Return processed image and label
#         """
#         image = Image.open(self.image_names[index])
#         image = self.transform(image)

#         label = self.labels[index]

#         return image, label
