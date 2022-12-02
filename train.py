import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.vqa_model import VQAModel
from dataset import VQADataset
import config


def get_dataloaders():
    # TODO: Add datasets and dataloaders for validation
    image_dir = os.path.join(config.DATASET_ROOT, 'train2014')
    questions_file = './questions_subset_train.pkl'
    questions_vocab_file = './questions_vocabulary_train.txt'
    answers_vocab_file = './answers_vocabulary_train.txt'
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(image_dir,
                               questions_file,
                               questions_vocab_file,
                               answers_vocab_file,
                               transform=image_transforms)
    print(len(train_dataset))
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16
    )
    dataloaders = {
        'train': train_dataloader,
    }
    return dataloaders


def train():
    # Define the model
    model = VQAModel(n_answers=100)
    model = model.to(config.DEVICE)

    optimizer_params = list(model.image_encoder.parameters()) + \
        list(model.question_encoder.parameters()) + \
        list(model.fc1.parameters()) + \
        list(model.fc2.parameters())

    # Define criterion, optimizer and lr_scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(optimizer_params, lr=1e-2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)

    dataloaders = get_dataloaders()

    for epoch in range(100):
        phases = ['train']
        for phase in phases:
            for batch_idx, (image, question, answer) in enumerate(dataloaders[phase]):
                # Move the inputs to cuda if available
                image = image.to(config.DEVICE)
                question = question.to(config.DEVICE)
                answer = answer.to(config.DEVICE)

                # Reset the optimizer
                optimizer.zero_grad()

                # Fetch the predictions
                prediction = model(image, question)
                # print(prediction)
                # print(answer)

                loss = criterion(prediction, answer)
                accuracy = sum(prediction.argmax(dim=1) == answer)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                if batch_idx % 1 == 0:
                    print(f"At {batch_idx} of {epoch}. {loss=}, {accuracy=}")
    return model


if __name__ == "__main__":
    train()
