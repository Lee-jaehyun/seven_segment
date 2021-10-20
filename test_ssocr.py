import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import cv2
from OCR_seven_ocr.source import models

from PIL import Image


# 인풋데이터 변환

def transform_image(path):
    mean = [0.80048384, 0.44734452, 0.50106468]
    std = [0.22327253, 0.29523788, 0.24583565]
    my_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
    image = Image.open(path).convert('RGB')
    return my_transform(image).unsqueeze(0)


new_model = 'resnet20' # resnet 구조 사용 가능, resnet8, resnet14, ... or resnet18, resnet34, resnet50 ...

new_model = models.__dict__[new_model](num_classes=10)

#print(new_model)

new_model.load_state_dict(torch.load('OCR_seven_ocr/source/model_state_dict.pt'))


for name, param in new_model.named_parameters():
    param.requires_grad = False
    print(name, ':', param.requires_grad)




class DigitData:
    def __init__(self, path, size=64, split='train'):
        self.path = path
        self.size = (size, size)
        
        # training set과 validation set 구분
        if split == 'train':
            self.image_files = open(os.path.join(path, 'train_data.txt'), 'r').read().splitlines()
      #  elif split == 'test':
      #      self.image_files = open(os.path.join(path, 'test_'))
        else:
            #self.image_files = open(os.path.join(path, 'test_data.txt'), 'r').read().splitlines()
            self.image_files = open(os.path.join(path, 'valid_data.txt'), 'r').read().splitlines()
            
        
        # 전체 데이터셋의 RGB 평균과 표준편차
        mean = [0.80048384, 0.44734452, 0.50106468]
        std = [0.22327253, 0.29523788, 0.24583565]
        self.transform = transforms.Compose([transforms.Resize(self.size), transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = os.path.join(self.path, self.image_files[idx])
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        target = int(self.image_files[idx].split('/')[0])
        return img, target


class AvgMeter(object):
    # 성능 측정을 위한 객체
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cal_acc(outputs, targets):
    # 정답률 확인을 위함
    _, pred = torch.max(outputs.data, 1)
    correct = pred.eq(targets).sum().item()
    acc = correct / targets.size(0)
    return acc

def train(model, data_loader, criterion, optimizer, cuda):
    '''
    모델의 학습을 위한 함수
    model: 학습에 사용되는 뉴럴 네트워크 모델
    data_loader: training data를 불러오는 객체
    criterion: loss function으로 cross entropy 사용
    optimizer: loss function에 따라 model의 parameter를 업데이트
    cuda: GPU 사용 여부
    '''
    model.train() # model을 training setting으로 교체

    # loss, accuracy 기록 (AvgMeter class 참고)
    train_loss = AvgMeter()
    train_acc = AvgMeter()
    for imgs, targets in data_loader:
        if cuda: # GPU를 사용하는 경우 data_loader의 출력을 CPU에서 GPU로 옮겨줌
            imgs, targets = imgs.cuda(), targets.cuda()
        outputs = model(imgs) # model 예측
        loss = criterion(outputs, targets) # loss 계산

        # model parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss, accuracy 기록
        train_loss.update(loss.item(), targets.size(0))
        train_acc.update(cal_acc(outputs, targets), targets.size(0))

    return train_loss.avg, train_acc.avg

def valid(model, data_loader, cuda):
    '''
    모델의 평가를 위한 함수
    model: 평가에 사용되는 뉴럴 네트워크 모델
    cuda: GPU 사용 여부
    '''
    model.eval() # model을 validation을 위한 상태로 변경

    valid_acc = AvgMeter() # accuracy 기록 (AvgMeter class 참고)
    predicted = 0

    for imgs, targets in data_loader:
        if cuda: # GPU를 사용하는 경우 data_loader의 출력을 CPU에서 GPU로 옮겨줌
            imgs, targets = imgs.cuda(), targets.cuda()
        with torch.no_grad(): # validation 시에 gradient 계산 불필요
            outputs = model(imgs) # model 예측
            _, predicted = torch.max(outputs.data, 1)

            
        valid_acc.update(cal_acc(outputs, targets), targets.size(0)) # accuracy 기록
        

    return valid_acc.avg, predicted


def predict_model(model, img):
    model.eval()
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
    return predicted
    




path = r'/home/ljh/Desktop/OCR_seven_ocr/dataset/digit_data' # digit_data가 있는 디렉토리
size = 64 # 이미지의 조정된 크기
batch_size = 128 # 한 번의 iteration에 사용할 instance의 수

train_data = DigitData(path, size, 'train')
valid_data = DigitData(path, size, 'valid')
test_data = DigitData(path, size, 'test')

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
print(train_loader)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


test_d = transform_image("seven_seg/스크린샷 2021-10-20 오전 9.00.27.png")

number = predict_model(new_model, test_d)
print(int(number))