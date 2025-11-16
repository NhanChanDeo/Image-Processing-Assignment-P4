import torch
from model.model import AgeGenderModel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # create model without downloading pretrained weights to avoid network issues
    model = AgeGenderModel(pretrained=False).to(device)
    model.eval()

    # create dummy input: batch size 2, 3 channels, 224x224
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        age_pred, gender_logit = model(x)

    print('age_pred shape:', age_pred.shape)
    print('gender_logit shape:', gender_logit.shape)

if __name__ == '__main__':
    main()
