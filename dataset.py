import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import ViTFeatureExtractor, BertTokenizer
from PIL import Image
import config
from utils.preprocess import Vocabulary, get_question_length


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

        if config.DEBUG:
            self.questions = self.questions[:100]

        self.max_question_length = get_question_length(max(
            self.questions.question,
            key=get_question_length)
        )
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
            # try:
            question_padded[:len(question)] = question
            # except Exception as e:
            # print(e)
            # print(question)
            # pass

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


class TransformerDataset(Dataset):
    def __init__(self,
                 image_dir_root,
                 questions_file,
                 answers_vocab_file,
                 transform=None,
                 phase='train',
                 return_raw=False):

        self.image_dir_root = image_dir_root
        self.questions_file = questions_file
        self.transform = transform
        self.phase = phase
        self.question_vocab = None
        self.return_raw = return_raw

        with open(answers_vocab_file, 'r') as f:
            self.answers_master = f.readlines()
        self.answers_master = [ans.strip() for ans in self.answers_master]
        self.answers_tokens = {ans: i for i, ans in
                               enumerate(self.answers_master)}

        self.questions = pd.read_pickle(questions_file)
        
        self.feature_extractor = ViTFeatureExtractor(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]).from_pretrained(
            'google/vit-base-patch16-224-in21k')
        self.question_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


        if config.DEBUG:
            self.questions = self.questions[:100]

        self.max_question_length = get_question_length(max(
            self.questions.question,
            key=get_question_length)
        )
        print(self.max_question_length)

    def __getitem__(self, item):
        row = self.questions.loc[item].to_dict()

        # Get the question and answers
        question = row['question']

        answer = row['most_picked_answer']

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
        
        image_features = self.feature_extractor(image,
                                                return_tensors='pt')
        
        image_features['pixel_values'] = image_features['pixel_values'].squeeze()
        
            
        
        question_features = self.question_tokenizer(question,
                                                    return_tensors='pt',
                                                    padding='max_length',
                                                    truncation=True,
                                                    max_length=24)

        for key in question_features.keys():
            question_features[key] = question_features[key].squeeze()
            
        if self.return_raw:        
            return image_features, question_features, self.answers_tokens[answer], image, question
        else:
            return image_features, question_features, self.answers_tokens[answer]

    def __len__(self):
        return len(self.questions)


if __name__ == '__main__':
    phase = 'val'
    dataset = 'rephrasings'
    image_dir = os.path.join(config.DATASET_ROOT, f'{phase}2014')
    questions_file = f'./questions_subset_{phase}_{dataset}.pkl'
    questions_vocab_file = f'./questions_vocabulary_{phase}_{dataset}.txt'
    answers_vocab_file = f'./answers_vocabulary_{phase}_{dataset}.txt'
    vqadataset = VQADataset(image_dir_root=image_dir,
                            tokenizer_source=questions_vocab_file,
                                    questions_file=questions_file,
                                    answers_vocab_file=answers_vocab_file,
                                    phase=phase)
    print(vqadataset.__getitem__(10))
