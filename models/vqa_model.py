import torch
import torch.nn as nn
from image_encoder import ImageEncoder
from question_encoder import (
    QuestionEncoderLSTM,
    QuestionEncoderTransformer
)


class VQAModel(nn.Module):
    def __init__(self,
                 image_encoder=ImageEncoder(),
                 question_encoder=QuestionEncoderLSTM(),
                 dropout_p=0.5,
                 fusion_hidden_units=1000,
                 n_answers=1000,
                 ):
        super(VQAModel, self).__init__()
        self.image_encoder = image_encoder
        self.question_encoder = question_encoder
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(self.image_encoder, fusion_hidden_units)
        self.fc2 = nn.Linear(fusion_hidden_units, n_answers)
        pass

    def forward(self, image, question):
        image_embedding = self.image_encoder(image)
        question_embedding = self.question_encoder(question)
        fused_features = torch.mul(image_embedding, question_embedding)
        fused_features = self.tanh(fused_features)
        fused_features = self.dropout(fused_features)
        fused_features = self.fc1(fused_features)
        fused_features = self.tanh(fused_features)
        fused_features = self.dropout(fused_features)
        fused_features = self.fc2(fused_features)
        return fused_features


if  __name__ == '__main__':
    pass
