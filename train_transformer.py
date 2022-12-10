import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.vqa_model import VQATransformer
from models.question_encoder import QuestionEncoderTransformer
from models.image_encoder import ImageEncoderTransformer
from dataset import TransformerDataset
import config


def get_dataloaders():
    dataloaders = {}
    phases = ['train', ]

    for phase in phases:
        image_dir = os.path.join(config.DATASET_ROOT, f'{phase}2014')
        questions_file = f'./questions_subset_{phase}.pkl'
        answers_vocab_file = f'./answers_vocabulary_train.txt'
        image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
        dataset = TransformerDataset(image_dir,
                                     questions_file,
                                     answers_vocab_file,
                                     transform=image_transforms,
                                     phase=phase)
        print(len(dataset))
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=4
        )
        dataloaders.__setitem__(
            phase, dataloader
        )

    return dataloaders


def train():
    image_encoder = ImageEncoderTransformer
    question_encoder = QuestionEncoderTransformer
    model = VQATransformer(
        image_encoder=image_encoder,
        question_encoder=question_encoder,
        n_answers=config.ANSWERS_VOCAB_SIZE,
    )
    model = model.to(config.DEVICE)

    # Set the parameters to optimize
    optimizer_params = list(model.fusion_layer.parameters()) + \
        list(model.classification_layer.parameters())

    # Define the criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)

    dataloaders = get_dataloaders()

    phases = ['train', ]

    metrics = {
        'train_metrics': {'loss': [],
                          'acc': []},
        'val_metrics': {'loss': [],
                        'acc': []}
    }

    for epoch in range(15):
        print(f"{epoch=}")
        for phase in phases:
            running_acc = 0
            for batch_idx, (image, question, answer) in tqdm(
                    enumerate(dataloaders[phase]),
                    total=len(dataloaders[phase])):
                # Move the inputs to cuda if available
                image = image#.to(config.DEVICE)
                question = question#.to(config.DEVICE)
                answer = answer#.to(config.DEVICE)

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
                  f"{running_acc / len(dataloaders[phase].dataset)}")

        # Save the model
        torch.save(model.state_dict(), os.path.join(config.MODEL_DIR,
                                                    f'cnn_lstm_{epoch}.pth'))

        metric_file = os.path.join(config.MODEL_DIR,
                                   'model.pkl')

        # Save metrics after each epoch
        with open(metric_file, 'wb') as file:
            pickle.dump(metrics, file)

    return model


if __name__ == '__main__':
    # dataloaders = get_dataloaders()
    # image, question, answer = next(iter(dataloaders['train']))
    # vqa = VQATransformer(image_encoder=ImageEncoderTransformer,
    #                      question_encoder=QuestionEncoderTransformer)
    # vqa(image, question)
    train()

