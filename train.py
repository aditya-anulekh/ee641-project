import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.vqa_model import VQAModel
from models.question_encoder import QuestionEncoderLSTM
from models.image_encoder import ImageEncoder
from dataset import VQADataset
import config

print(f"Using {config.DEVICE}")


def get_dataloaders():
    # TODO: Add datasets and dataloaders for validation

    dataloaders = {}
    phases = ['train', 'val']
    
    for phase in phases:

        image_dir = os.path.join(config.DATASET_ROOT, f'{phase}2014')
        questions_file = f'./questions_subset_{phase}.pkl'
        questions_vocab_file = f'./questions_vocabulary_train.txt'
        answers_vocab_file = f'./answers_vocabulary_train.txt'
        image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),        
        ])
        dataset = VQADataset(image_dir,
                            questions_file,
                            questions_vocab_file,
                            answers_vocab_file,
                            transform=image_transforms,
                            phase=phase)
        print(len(dataset))
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=64
        )
        dataloaders.__setitem__(
            phase, dataloader
        )

    return dataloaders


def train():
    # Define the model
    image_encoder = ImageEncoder(pretrained=True)
    question_encoder = QuestionEncoderLSTM(config.QUESTION_VOCAB_SIZE)
    model = VQAModel(
        image_encoder=image_encoder,
        question_encoder=question_encoder,
        n_answers=config.ANSWERS_VOCAB_SIZE
    )
    model = model.to(config.DEVICE)

    optimizer_params = list(model.image_encoder.fc.parameters()) + \
        list(model.question_encoder.parameters()) + \
        list(model.fc1.parameters()) + \
        list(model.fc2.parameters())

    # Define criterion, optimizer and lr_scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)

    dataloaders = get_dataloaders()

    phases = ['train', 'val']

    metrics = {
        'train_metrics': {'loss': [],
                          'acc': []},
        'val_metrics': {'loss': [],
                        'acc': []}
    }

    for epoch in range(30):
        print(f"{epoch=}")
        for phase in phases:
            running_acc = 0
            for batch_idx, (image, question, answer) in tqdm(enumerate(dataloaders[phase]), 
                                                             total=len(dataloaders[phase])):
                # Move the inputs to cuda if available
                image = image.to(config.DEVICE)
                question = question.to(config.DEVICE)
                answer = answer.to(config.DEVICE)

                # Reset the optimizer
                optimizer.zero_grad()
                
                # Set model in training/validation mode according to the phase
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                # Fetch the predictions
                prediction = model(image, question)

                # Calculate loss and backprop
                with torch.set_grad_enabled(phase == "train"):
                    loss = criterion(prediction, answer)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_acc += sum(prediction.argmax(dim=1) == answer)
            print(f"{phase}: At {batch_idx} of {epoch}. Accuracy = "
                  f"{running_acc/len(dataloaders[phase].dataset)}")

        # Save the model
        torch.save(model.state_dict(), os.path.join(config.MODEL_DIR,
                                                    f'cnn_lstm_{epoch}.pth'))
        # Save metrics after each epoch
        with open(f'metrics_{epoch}.pkl', 'wb') as file:
            pickle.dump(metrics, file)

    return model


if __name__ == "__main__":
    train()
