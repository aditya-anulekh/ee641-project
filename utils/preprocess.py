import os
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
               'question_type': annotations_data['annotations'][i]['question_type'],
               'question': questions_data['questions'][i]['question'],
               'most_picked_answer': annotations_data['annotations'][i][
                   'multiple_choice_answer'],
               'answers': answers_list}

        data.append(row)
    input_df = pd.DataFrame(data)
    input_df.to_pickle('../questions.pkl')
    return input_df


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
