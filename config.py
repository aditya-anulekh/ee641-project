"""
Configuration file to store global variables
"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = True
LEARNING_RATE = 2e-4
DATASET_ROOT = '/study/1 USC/3 Fall 2022/EE641/Project/VQA_dataset/'
QUESTION_VOCAB_SIZE = 2420
