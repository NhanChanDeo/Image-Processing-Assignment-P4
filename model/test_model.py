import torch

from model.model import AgeGenderModel
from model.losses import MultiTaskLoss

from preprocess.dataloader import build_loaders

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    train_loader, _, _ = build_loaders(
    train_csv="labeled/train/label.csv",
    train_img="labeled/train/img",
    batch_size=4
)

    model = AgeGenderModel(pretrained=True).to(device)
    criterion = MultiTaskLoss(alpha=1.0, beta=0.2)

   # 1. Nhận 2 giá trị mà loader trả về (ảnh và dictionary samples)
    images, samples = next(iter(train_loader))

    # 2. Tải dữ liệu lên device (GPU/CPU)
    images = images.to(device)
    
    # 3. Unpack (gỡ) dictionary samples và đưa lên device
    ages = samples['age'].to(device)
    genders = samples['gender'].to(device).float()

    age_pred, gender_pred = model(images)
    loss, la, lg = criterion(age_pred, ages, gender_pred, genders)

    print("age_pred:", age_pred)
    print("gender_pred:", gender_pred)
    print("Loss:", loss.item())
    print("Age loss:", la.item())
    print("Gender loss:", lg.item())

if __name__ == "__main__":
    test()
