# TODO: Fix deprecation warning

import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models import vgg19
from transformers import (
    ViTFeatureExtractor,
    ViTModel
)
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
        # with torch.no_grad():
        x = self.base_model(x)
        x = self.fc(x)

        # Calculate the norm and divide the vector
        l2_norm = x.norm(p=2, dim=1, keepdim=True).detach()
        x = x.div(l2_norm)
        return x


class ImageEncoderTransformer(nn.Module):
    def __init__(self):
        super(ImageEncoderTransformer, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            'google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k')

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor([i for i in x],
                                       return_tensors='pt')
            x = self.model(**x)

        return x


if __name__ == "__main__":
    img_encoder = ImageEncoderTransformer()
    x = torch.rand((3, 3, 224, 224))
    print(img_encoder(x))
    pass

