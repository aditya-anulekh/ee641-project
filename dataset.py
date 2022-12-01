import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import config
from utils.preprocess import Vocabulary


class VQADataset(Dataset):
    def __init__(self,
                 image_dir_root,
                 questions_file,
                 tokenizer_source,
                 answers_vocab_file,
                 transform=None,
                 tokenizer=Vocabulary,
                 phase='train'):
        self.image_dir_root = image_dir_root
        self.questions_file = questions_file
        self.transform = transform
        self.phase = phase
        self.question_vocab = None
        if tokenizer:
            self.question_vocab = tokenizer(tokenizer_source)
        with open(answers_vocab_file, 'r') as f:
            self.answers_master = f.readlines()
        self.answers_master = [ans.strip() for ans in self.answers_master]
        self.answers_tokens = {ans: i for i, ans in
                               enumerate(self.answers_master)}

        self.questions = pd.read_pickle(questions_file)

        # Crop the dataset for debugging purposes
        if config.DEBUG:
            self.questions = self.questions[:10]

        self.max_question_length = len(max(self.questions.question,
                                       key=lambda x: len(x.split())).split())
        print(self.max_question_length)
        pass

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        row = self.questions.loc[item].to_dict()

        # Get the question and answers
        question = row['question']
        # print(question)
        if self.question_vocab:
            question = torch.LongTensor(self.question_vocab(question))
            question_padded = torch.zeros(self.max_question_length,
                                          dtype=question.dtype)
            try:
                question_padded[:len(question)] = question
            except Exception as e:
                # print(e)
                # print(question)
                pass

        answer = row['most_picked_answer']
        # print(answer)
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
        return image, question_padded, self.answers_tokens[answer]


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
