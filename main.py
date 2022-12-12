import sys
from io import BytesIO
from PIL import Image
import torch
from torchvision.transforms import transforms
from utils.preprocess import Vocabulary
from models.vqa_model import VQAModel
from models.question_encoder import QuestionEncoderLSTM
from models.image_encoder import ImageEncoder
import config


def predict(image_url, question):
    image_encoder = ImageEncoder(pretrained=True)
    question_encoder = QuestionEncoderLSTM(config.QUESTION_VOCAB_SIZE)
    answers_master = None
    
    with open('./answers_vocabulary_val_rephrasings.txt', 'r') as f:
        answers_master = f.readlines()
        answers_master = [ans.strip() for ans in answers_master]


    model = VQAModel(
        image_encoder=image_encoder,
        question_encoder=question_encoder,
        n_answers=config.ANSWERS_VOCAB_SIZE
    )

    print(model.load_state_dict(
        torch.load('./saved_models/rephrasings/cnn_lstm_14.pth')
    ))

    model = model.to(config.DEVICE)
    
    model.eval()

    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    # Apply transforms on the image
    # image = requests.get(image_url)
    image_og = Image.open(image_url)
    image = image_transforms(image_og)
    image = image.to(config.DEVICE)

    # Encode the question
    vocab = Vocabulary(
        vocabulary_file=f'./questions_vocabulary_val_rephrasings.txt'
    )

    # Pad the question
    question = torch.LongTensor(vocab(question))
    question_padded = torch.zeros(27, dtype=question.dtype)
    question_padded[:len(question)] = question
    question = question.to(config.DEVICE)

    output = model(
        torch.unsqueeze(image, 0),
        torch.unsqueeze(question, 0)
    )
    
    top_ans = torch.argsort(output, dim=1, descending=True)

    answer_pred = [answers_master[int(top_ans[0][k])] for k in range(3)]

    return answer_pred, image_og


if __name__ == '__main__':
    # Get arguments from command line

    if len(sys.argv) != 3:
        print("Please provide exactly the path and question")
        sys.exit()
    
    image_url = sys.argv[1]
    question = sys.argv[2]

    out = predict(
        image_url,
        question
    )    

    print(out)
    