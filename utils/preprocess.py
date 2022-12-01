import argparse
import os
import re
import json
import pandas as pd
from tqdm import tqdm


# from .. import config


def generate_questions_csv(questions_file, annotations_file, phase):
    f1 = open(questions_file)
    questions_data = json.load(f1)

    f2 = open(annotations_file)
    annotations_data = json.load(f2)

    data = []

    for i in tqdm(range(len(questions_data['questions']))):
        answers_list = []
        # Collect all answers for that question
        for j in range(len(annotations_data['annotations'][i]['answers'])):
            answers_list.append(
                annotations_data['annotations'][i]['answers'][j]['answer'])

        # Create a dictionary to store questions and answers
        row = {'image_id': questions_data['questions'][i]['image_id'],
               'question_id': questions_data['questions'][i]['question_id'],
               'question_type': annotations_data['annotations'][i][
                   'question_type'],
               'question': questions_data['questions'][i]['question'],
               'most_picked_answer': annotations_data['annotations'][i][
                   'multiple_choice_answer'],
               'answers': answers_list}

        data.append(row)
    input_df = pd.DataFrame(data)
    path_to_output = os.path.join('..', f'questions_{phase}.pkl')
    input_df.to_pickle(path_to_output)
    return input_df, path_to_output


def generate_questions_vocab(dataset_df: pd.DataFrame, phase):
    questions = dataset_df.question.tolist()
    vocabulary = set()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

    for i, question in enumerate(questions):
        words = SENTENCE_SPLIT_REGEX.split(question)
        words = [w.strip() for w in words if len(w.strip()) > 0]
        vocabulary.update(words)

    vocabulary = list(vocabulary)
    vocabulary.sort()

    # Insert unknown and pad tokens into the vocabulary
    vocabulary.insert(0, '<pad>')
    vocabulary.insert(1, '<unk>')

    with open(f'../questions_vocabulary_{phase}.txt', 'w') as f:
        f.writelines([f"{w}\n" for w in vocabulary])
    return vocabulary


def generate_answers_vocab(dataset_df: pd.DataFrame,
                           num_answers: int,
                           phase: str):
    top_answers = dataset_df.most_picked_answer.value_counts().nlargest(
        num_answers).index
    top_answers = top_answers.tolist()
    top_answers.sort()
    top_answers.insert(0, '<unk>')
    top_answers = top_answers[:num_answers - 1]
    with open(f'../answers_vocabulary_{phase}.txt', 'w') as f:
        f.writelines([f"{ans}\n" for ans in top_answers])
    return top_answers


def get_answers_subset(questions_file: str, question_contents: str, phase: str):
    questions = pd.read_pickle(questions_file)
    questions = questions.loc[
        questions.question_type.str.contains(question_contents) &
        questions.question.str.contains(question_contents)
    ]
    questions.reset_index(drop=True, inplace=True)
    questions.to_pickle(f"../questions_subset_{phase}.pkl")
    return questions


class Vocabulary:
    def __init__(self, vocabulary_file):
        self.vocabulary_file = vocabulary_file
        self.vocabulary = self.load_lines(vocabulary_file)
        self.word_indices = {word:i for i, word in enumerate(self.vocabulary)}
        self.vocab_size = len(self.vocabulary)

    def idx2word(self, idx):
        return self.vocabulary[idx]

    def word2idx(self, word):
        if word in self.word_indices:
            return self.word_indices[word]
        else:
            return self.word_indices['<unk>']

    def tokenize_input(self, sentence):
        SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
        sentence = SENTENCE_SPLIT_REGEX.split(sentence)
        words = [w.strip() for w in sentence if len(w.strip()) > 0]
        return [self.word2idx(word) for word in words]

    @staticmethod
    def load_lines(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add argument to select answer filter
    parser.add_argument('--filter', type=str, default='color',
                        help='Filter answers based on this')

    args = parser.parse_args()

    phases = ['train']

    for phase in phases:
        _, output_file = generate_questions_csv(
            os.path.join(
                '/study/1 USC/3 Fall 2022/EE641/Project/VQA_dataset/',
                'v2_Questions_Train_mscoco',
                'v2_OpenEnded_mscoco_train2014_questions.json'),
            os.path.join(
                '/study/1 USC/3 Fall 2022/EE641/Project/VQA_dataset/',
                'v2_Annotations_Train_mscoco',
                'v2_mscoco_train2014_annotations.json'),
            phase
        )
        data = get_answers_subset(output_file, args.filter, phase)
        _ = generate_questions_vocab(data, phase)
        _ = generate_answers_vocab(data, 100, phase)
