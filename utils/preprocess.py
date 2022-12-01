import os
import re
import json
import pandas as pd
from tqdm import tqdm


# from .. import config


def generate_questions_csv(questions_file, annotations_file):
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
    input_df.to_pickle('../questions.pkl')
    return input_df


def generate_questions_vocab(dataset_df: pd.DataFrame):
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

    with open('../questions_vocabulary.txt', 'w') as f:
        f.writelines([f"{w}\n" for w in vocabulary])
    return vocabulary


def generate_answers_vocab(dataset_df: pd.DataFrame,
                           num_answers: int):
    top_answers = dataset_df.most_picked_answer.value_counts().nlargest(
        num_answers).index
    top_answers = top_answers.tolist()
    top_answers.sort()
    top_answers.insert(0, '<unk>')
    top_answers = top_answers[:num_answers - 1]
    with open('../answers_vocabulary.txt', 'w') as f:
        f.writelines([f"{ans}\n" for ans in top_answers])
    return top_answers


def get_answers_subset(questions_file: str, question_contents: str):
    questions = pd.read_pickle(questions_file)
    return questions.loc[
        questions.question_type.str.contains(question_contents) &
        questions.question.str.contains(question_contents)
    ]


class Vocabulary:
    def __init__(self, vocabulary_file):
        self.vocabulary_file = vocabulary_file

        pass


if __name__ == "__main__":
    data = generate_questions_csv(
        os.path.join(
            '/study/1 USC/3 Fall 2022/EE641/Project/VQA_dataset/',
            'v2_Questions_Train_mscoco',
            'v2_OpenEnded_mscoco_train2014_questions.json'),
        os.path.join(
            '/study/1 USC/3 Fall 2022/EE641/Project/VQA_dataset/',
            'v2_Annotations_Train_mscoco',
            'v2_mscoco_train2014_annotations.json'),
    )
    print(data)
