import torch
import torch.nn as nn
from .image_encoder import (
    ImageEncoder,
    ImageEncoderTransformer
)
from .question_encoder import (
    QuestionEncoderLSTM,
    QuestionEncoderTransformer
)


class VQAModel(nn.Module):
    def __init__(self,
                 image_encoder,
                 question_encoder,
                 dropout_p=0.5,
                 fusion_hidden_units=1000,
                 n_answers=1000,
                 ):
        super(VQAModel, self).__init__()
        self.image_encoder = image_encoder
        self.question_encoder = question_encoder
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(self.image_encoder.embedding_size,
                             fusion_hidden_units)
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


class VQATransformer(nn.Module):
    def __init__(self,
                 image_encoder,
                 question_encoder,
                 n_answers=1000,
                 intermediate_layer_size=1000,
                 dropout=0.5):
        super(VQATransformer, self).__init__()
        self.image_encoder = image_encoder()
        self.question_encoder = question_encoder()
        self.n_answers = n_answers
        self.intermediate_layers_size = intermediate_layer_size
        self.dropout = dropout
        fusion_size = self.image_encoder.model.config.hidden_size + \
            self.question_encoder.model.config.hidden_size

        self.fusion_layer = nn.Linear(fusion_size, intermediate_layer_size)
        self.classification_layer = nn.Linear(intermediate_layer_size,
                                              self.n_answers)

    def forward(self, image, question):
        image_features = self.image_encoder(image)['pooler_output']
        question_features = self.question_encoder(question)['pooler_output']
        fusion_features = torch.cat([
            image_features,
            question_features
        ], dim=1)
        fusion_features = self.fusion_layer(fusion_features)
        fusion_features = self.classification_layer(fusion_features)
        return fusion_features


if __name__ == '__main__':
    vqa = VQATransformer(ImageEncoderTransformer,
                         QuestionEncoderTransformer)
    output = vqa([torch.rand(3, 224, 224), torch.rand(3, 224, 224)],
                 ["This is a test question", "This is another test question"])
    print(output.shape)

    pass
