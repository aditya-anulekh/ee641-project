# TODO: Fix deprecation warning

import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models import vgg19
import config


class ImageEncoder(nn.Module):
    def __init__(self, base_model=vgg19, embedding_size=1024, **kwargs):
        super(ImageEncoder, self).__init__()
        base_model = base_model(**kwargs)
        self.embedding_size = embedding_size

        # Get the number of input features to the classifier layer
        in_features = base_model.classifier[-1].in_features

        # Set the output layer to None
        base_model.classifier = nn.Sequential(
            *list(base_model.classifier.children())[:-1]
        )
        self.base_model = base_model
        self.fc = nn.Linear(in_features, embedding_size)

    def forward(self, x):
        # Extract features using the base model
        with torch.no_grad():
            x = self.base_model(x)
        x = self.fc(x)

        # Calculate the norm and divide the vector
        l2_norm = x.norm(p=2, dim=1, keepdim=True).detach()
        x = x.div(l2_norm)
        return x


if __name__ == "__main__":
    img_encoder = ImageEncoder(pretrained=True)
    print(summary(img_encoder.to(config.DEVICE), (3, 224, 224)))
    pass

