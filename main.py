import torch
from torch import nn
from torchvision import transforms, datasets
import torchvision
from torch.utils.data import DataLoader
import os
from pathlib import Path
from flask import Flask, render_template, request, redirect
import requests
import zipfile
from PIL import Image
data_path = Path('scans/')
image_path = data_path / 'x'
if not data_path.is_dir():
    image_path.mkdir(exist_ok =True, parents=True)
else:
    print('already created')
with open(data_path / "scan.zip", 'wb') as f:
  request = requests.get('https://github.com/ColinJ69/ScaniAi/raw/main/scans.zip')
  f.write(request.content)
with zipfile.ZipFile(data_path / "scan.zip", "r") as zip_ref:
  zip_ref.extractall(image_path)
os.remove(data_path/'scan.zip')

auto_data_transforms = transforms.Compose([transforms.Resize((80,80)), transforms.ToTensor()])

train_image_fold = datasets.ImageFolder(root=image_path, transform = auto_data_transforms, target_transform=None)

train_dataloader = DataLoader(train_image_fold,batch_size=8, num_workers = 0, shuffle = True)
class_names = train_image_fold.classes



from torchvision.models import resnet50, ResNet50_Weights
resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet50(weights="IMAGENET1K_V1")
resnet50(pretrained=True)
resnet50(True)


weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.classifier = torch.nn.Sequential(nn.Dropout(0.2, inplace = True), nn.Linear(in_features = 1280, out_features = len(class_names), bias=True))

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(params = model.parameters(), lr=0.001)
def accuracy(y_pred, y_true):
    correct = torch.eq(y_pred, y_true).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
def train():
    loss, acc = 0,0
    model.train()

    for batch, (x,y) in enumerate(train_dataloader):

        y_pred = model(x)
        lossy = loss_fn(y_pred, y)
        accy = accuracy(y_pred = y_pred.argmax(dim=1), y_true = y)
        loss += lossy
        acc += accy
        optim.zero_grad()
        lossy.backward()
        optim.step()
    loss /= len(train_dataloader)
    acc /= len(train_dataloader)
    print(f'LOSS : {loss:.5f} || ACC : {acc:.2f}')
    torch.save(model, 'model.pth')
def create(user_img):
  input_img = data_path / "input_img"
  if input_img.is_dir() & (input_img / "input_img.zip").is_dir():
    print('restart or smth')
  else:
    input_img.mkdir(parents=True, exist_ok = True)
    with open(data_path / "input_img" / "input_img.zip", 'wb') as f:
      request = requests.get(user_img)
      f.write(request.content)
  return input_img / 'input_img.zip'
def test(user_img):
    entire_model = torch.load('model.pth')
    entire_model.eval()
    img = Image.open(create(user_img))
    if img.mode == 'RGB':
      img = img.convert('L')

    with torch.inference_mode():
      img_trans = auto_data_transforms(img).unsqueeze(dim=0)
      output = entire_model(img_trans)

    predicted_class = torch.argmax(torch.softmax(output, dim=1),dim=1)
    
    print(class_names[predicted_class.item()])


def tr_results():
    epochs = 5

    for e in range(epochs):
      train()
def te_results(user_img):
  test(user_img)

if __name__ == '__main__':
    app = Flask(__name__)
    @app.route('/')
    def welcome_page():
        return Flask.render_templatehomepage.html')
    @app.route('/result', methods = ['GET','POST'])
    def result():
      pic = request.form['photo']
      return render_template('result.html', breed=te_results(pic))
    app.run(debug=True)
