import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import time
import os
import torchvision.models as models
from tqdm import tqdm
from typing import Literal
from functools import reduce
import csv

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def load_patients(csv_path, data_dir_path):
    patients = {}
    with open(csv_path, encoding= 'unicode_escape') as csvFile :
        csvDictReader = csv.DictReader(csvFile)
        for row in csvDictReader:
            pid = row["Patient ID"]
            if patients.get(pid) is None:
                patients[pid] = []
            patients[pid].append(os.path.join(data_dir_path, row["File name"]))

    return [patient for patient in patients.values()]

def percent_list_slice(x, start=0., end=1.):
    return x[int(len(x)*start):int(len(x)*end)]

class CovidCT(Dataset):
    def __init__(self,
                 data_root,
                 mode: Literal["train", "valid", "test"] = "train",
                 transform=None):
        if mode == "train":
            start, end = 0.0, 0.6
        elif mode == "valid":
            start, end = 0.6, 0.8
        elif mode == "test":
            start, end = 0.8, 1.0

        normal_patients = load_patients(
            os.path.join(data_root, "meta_data_normal.csv"),
            os.path.join(data_root, "curated_data/curated_data/1NonCOVID"))
        normal_patients = percent_list_slice(normal_patients, start, end)
        normal_file_paths = reduce(lambda a, b: a+b, normal_patients)

        covid_patients = load_patients(
            os.path.join(data_root, "meta_data_covid.csv"),
            os.path.join(data_root, "curated_data/curated_data/2COVID"))
        covid_patients = percent_list_slice(covid_patients, start, end)
        covid_file_paths = reduce(lambda a, b: a+b, covid_patients)

        self.file_paths = normal_file_paths + covid_file_paths
        self.labels = [0]*len(normal_file_paths) + [1]*len(covid_file_paths)
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        image = Image.open(self.file_paths[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, self.labels[index]

CUDA_DEVICES = 
init_lr = 0.001

training_acc_list = []
training_loss_list = []

label_list = []
pred_list = []

seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)
torch.cuda.manual_seed(seed_value)

# Save model every 5 epochs
checkpoint_interval = 5
if not os.path.isdir('./Checkpoint/4/'):
    os.mkdir('./Checkpoint/4/')


# Setting learning rate operation
def adjust_lr(optimizer, epoch):
    # 1/10 learning rate every 5 epochs
    lr = init_lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    # If out of memory , adjusting the batch size smaller
    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    trainset = CovidCT("./input/", "train", data_transform)
    train_dl = DataLoader(trainset, batch_size=9, shuffle=True, num_workers=3)
    validset = CovidCT("./input/", "valid", data_transform)
    valid_dl = DataLoader(validset, batch_size=9, shuffle=False, num_workers=3)
    classes = ['1NonCOVID','2COVID']

    model=models.resnet18(pretrained=True)
    model.fc=nn.Linear(in_features=512, out_features=2, bias=True) #如果要使用預訓練模型，記得修改最後一層輸出的class數量
    #print(model)
    print("==========")

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    model = model.cuda(CUDA_DEVICES)

    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Training epochs
    num_epochs = 30
    criterion = nn.CrossEntropyLoss()

    # Optimizer setting
    optimizer = torch.optim.SGD(params=model.parameters(), lr=init_lr, momentum=0.9)

    # Log
    with open('TrainingAccuracy.txt','w') as fAcc:
        print('Accuracy\n', file = fAcc)
    with open('TrainingLoss.txt','w') as fLoss:
        print('Loss\n', file = fLoss)

    for epoch in range(num_epochs):
        model.train()
        localtime = time.asctime( time.localtime(time.time()) )
        print('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,num_epochs,localtime))
        print('-' * len('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,num_epochs,localtime)))

        training_loss = 0.0
        training_corrects = 0
        adjust_lr(optimizer, epoch)

        for i, (inputs, labels) in (enumerate(tqdm(train_dl))):

            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += float(loss.item() * inputs.size(0))
            training_corrects += torch.sum(preds == labels.data).item()

        training_loss = training_loss / len(trainset)
        training_acc = training_corrects /len(trainset)
        print('\n Training loss: {:.4f}\taccuracy: {:.4f}\n'.format(training_loss,training_acc))


        # Check best accuracy model ( but not the best on test )
        if training_acc > best_acc:
            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())


        with open('TrainingAccuracy.txt','a') as fAcc:
            print('{:.4f} '.format(training_acc), file = fAcc)
            training_acc_list.append(training_acc)
            print(training_acc_list)

        with open('TrainingLoss.txt','a') as fLoss:
            print('{:.4f} '.format(training_loss), file = fLoss)
            training_loss_list.append(training_loss)
            print(training_loss_list)
            print()

        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(model, './Checkpoint/4/model-epoch-{:d}-train.pth'.format(epoch + 1))

        model = model.cuda(CUDA_DEVICES)
        model.eval()
        total_correct = 0
        total = 0
        class_correct = list(0. for i in enumerate(classes))
        class_total = list(0. for i in enumerate(classes))

        with torch.no_grad():
            for inputs, labels in tqdm(valid_dl):
                inputs = Variable(inputs.cuda(CUDA_DEVICES))
                labels = Variable(labels.cuda(CUDA_DEVICES))
                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()


                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

            for i, c in enumerate(classes):
              if(class_total[i]==0):
                print('Accuracy of %5s : %8.4f %%' % (
                c, 100 * 0))
              else:
                print('Accuracy of %5s : %8.4f %%' % (
                c, 100 * class_correct[i] / class_total[i]))

            # Accuracy
            print('\nAccuracy on the ALL val images: %.4f %%'
              % (100 * total_correct / total))

    # Save best training/valid accuracy model ( not the best on test )
    model.load_state_dict(best_model_params)
    best_model_name = './Checkpoint/4/model-{:.2f}-best_train_acc.pth'.format(best_acc)
    torch.save(model, best_model_name)
    print("Best model name : " + best_model_name)

    # plot the train acc curve
    plt.plot(training_acc_list)
    plt.title('Training Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.show()
    plt.savefig('acc_4.png')
    plt.cla()
    plt.clf()
    plt.close()

    # plot the train loss curve
    plt.plot(training_loss_list)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.show()
    plt.savefig('loss_4.png')
    plt.cla()
    plt.clf()
    plt.close()

    return model

def test(train_model):
    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    testset = CovidCT("./input/", "test", data_transform)
    test_dl = DataLoader(testset, batch_size=7, shuffle=False, pin_memory=True, num_workers=3)
    classes = ['1NonCOVID','2COVID']

    # Load model
    model = train_model
    model = model.cuda(CUDA_DEVICES)
    model.eval()

    total_correct = 0
    total = 0
    class_correct = list(0. for i in enumerate(classes))
    class_total = list(0. for i in enumerate(classes))

    with torch.no_grad():
        for inputs, labels in tqdm(test_dl):
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # totoal
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()

            # batch size
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

                label_list.append(labels[i].item())
                pred_list.append(predicted[i].item())

    for i, c in enumerate(classes):
        print('Accuracy of %5s : %8.4f %%' % (
        c, 100 * class_correct[i] / class_total[i]))

    # Accuracy
    print('\nAccuracy on the ALL test images: %.4f %%'
      % (100 * total_correct / total))

    y_true = label_list
    y_pred = pred_list

    # compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['NonCOVID', 'COVID'],
           yticklabels=['NonCOVID', 'COVID'],
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    #plt.show()
    plt.savefig('CM_4.png')
    plt.cla()
    plt.clf()
    plt.close()

    f1score = f1_score(y_true, y_pred)
    print("\nF1 Score: %.4f %%" % (100 * f1score))

    precision = precision_score(y_true, y_pred)
    print("\nPrecision: %.4f %%" % (100 * precision))

    recall = recall_score(y_true, y_pred)
    print("\nRecall: %.4f %%" % (100 * recall))

    auc = roc_auc_score(y_true, y_pred)
    print("\nAUC: %.4f %%\n" % (100 * auc))

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    #plt.show()
    plt.savefig('ROC_4.png')
    plt.cla()
    plt.clf()
    plt.close()

if __name__ == '__main__':
    train_model = train()
    test(train_model)




































