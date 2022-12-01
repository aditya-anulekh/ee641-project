import torch
import torch.nn as nn
from torchsummary import summary
from transformers import BertTokenizer, BertModel


class QuestionEncoderLSTM(nn.Module):
    def __init__(self,
                 question_vocab_size,
                 word_embedding_size=300,
                 question_embedding_size=1024,
                 num_lstm_layers=2,
                 num_hidden_units=512,
                 ):
        super(QuestionEncoderLSTM, self).__init__()
        # TODO: Fix embedding size
        self.embedding = nn.Embedding(question_vocab_size, word_embedding_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=word_embedding_size,
                            hidden_size=num_hidden_units,
                            num_layers=num_lstm_layers)
        self.fc = nn.Linear(2 * num_lstm_layers * num_hidden_units,
                            question_embedding_size)

    def forward(self, question):
        question_vector = self.embedding(question)
        question_vector = self.tanh(question_vector)
        question_vector = question_vector.transpose(0, 1)
        _, (hidden_state, cell_state) = self.lstm(question_vector)
        question_feature = torch.cat((hidden_state, cell_state), 2)
        question_feature = question_feature.transpose(0, 1)
        question_feature = question_feature.reshape(question_feature.size()[0],
                                                    -1)
        question_feature = self.tanh(question_feature)
        question_feature = self.fc(question_feature)
        return question_feature


class QuestionEncoderTransformer(nn.Module):
    def __init__(self):
        super(QuestionEncoderTransformer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x):
        with torch.no_grad():
            x = self.tokenizer(x, return_tensors='pt')
            print(x)
            x = self.model(**x, return_dict=True)
        return x


if __name__ == '__main__':
    qe = QuestionEncoderLSTM(2000)
    # print(qe('This is a test sentence'))
    print(qe(torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])).shape)
    pass
