import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms, io
from PIL import Image
import config
from utils.preprocess import Vocabulary


class VQADataset(Dataset):
    def __init__(self,
                 image_dir_root,
                 questions_file,
                 questions_vocab_file,
                 answers_vocab_file,
                 transform=None,
                 phase='train'):
        self.image_dir_root = image_dir_root
        self.questions_file = questions_file
        self.transform = transform
        self.phase = phase
        self.question_vocab = Vocabulary(questions_vocab_file)
        with open(answers_vocab_file, 'r') as f:
            self.answers_master = f.readlines()
        self.answers_master = [ans.strip() for ans in self.answers_master]

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
        print(question)
        question = self.question_vocab.tokenize_input(question)
        answer = row['most_picked_answer']
        print(answer)
        if answer not in self.answers_master:
            answer = '<unk>'

        # Get the path of the image corresponding to the question
        image = os.path.join(
            self.image_dir_root,
            f'COCO_{self.phase}2014_{str(row["image_id"]).zfill(12)}.jpg'
        )

        # Convert image to RGB
        image = Image.open(image).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, question, answer


if __name__ == '__main__':
    image_dir = os.path.join(config.DATASET_ROOT, 'train2014')
    questions_file = './questions_subset_train.pkl'
    questions_vocab_file = './questions_vocabulary_train.txt'
    answers_vocab_file = './answers_vocabulary_train.txt'
    vqadataset = VQADataset(image_dir,
                            questions_file,
                            questions_vocab_file,
                            answers_vocab_file)
    print(vqadataset.__getitem__(10))
