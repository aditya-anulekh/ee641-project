"""
Configuration file to store global variables
"""

import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = True
LEARNING_RATE = 2e-4
DATASET_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
QUESTION_VOCAB_SIZE = 2420
ANSWERS_VOCAB_SIZE = 100
MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models')