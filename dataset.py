import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms, io
from PIL import Image
import config


class VQADataset(Dataset):
    def __init__(self,
                 image_dir_root,
                 questions_file,
                 transform=None,
                 phase='train'):
        self.image_dir_root = image_dir_root
        self.questions_file = questions_file
        self.transform = transform
        self.phase = phase

        self.questions = pd.read_pickle(questions_file)

        # Crop the dataset for debugging purposes
        if config.DEBUG:
            self.questions = self.questions[:100]

        pass

    def __len__(self):
        return len(self.questions_file)

    def __getitem__(self, item):
        row = self.questions.loc[item].to_dict()

        # Get the question and answers
        question = row['question']
        answer = row['most_picked_answer']

        # Get the path of the image corresponding to the question
        image = os.path.join(
            self.image_dir_root,
            f'COCO_{self.phase}2014_{str(row["image_id"]).zfill(12)}.jpg'
        )

        # Convert image to RGB
        image = np.array(Image.open(image).convert("RGB"))

        if self.transform:
            image = self.transform(image)
        return image, question, answer


if __name__ == '__main__':
    image_dir = os.path.join(config.DATASET_ROOT, 'train2014')
    questions_file = './questions.pkl'
    vqadataset = VQADataset(image_dir, questions_file)
    print(vqadataset.__getitem__(10))
