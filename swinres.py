import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.utils.data.dataloader
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision.models.swin_transformer import swin_t
from torchvision.models import resnet50, ResNet50_Weights
from dataset import CustomDataset


class SwinResModel(nn.Module):
    def __init__(self, out_features=5):
        super().__init__()

        self.swinModel = swin_t(weights=None)
    
        swin_in_features = self.swinModel.head.in_features
        self.swinModel.head = nn.Linear(swin_in_features, 150)
                                   
        self.resModel = resnet50(weights=None)
     
        res_in_features = self.resModel.fc.in_features
        self.resModel.fc = nn.Linear(res_in_features, 150)     

        self.fc = nn.Sequential(
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(300, out_features)
        )     
        

    def forward(self, x):
        out_swin = self.swinModel(x)
        out_res = self.resModel(x)
        cat = torch.cat((out_swin, out_res), 1)
        out = self.fc(cat)
        return out


def visualizeResult(train_loss, train_acc, val_loss, val_acc, epochs):
    plt.title("Model's Loss Visualization")
    plt.plot(range(epochs), train_loss, label = "training loss")
    plt.plot(range(epochs), val_loss, label = "validation loss")
    plt.legend()
    plt.xlabel ("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Model's Accuracy Visualization")
    plt.plot(range(epochs), train_acc, label="training accuracy")
    plt.plot(range(epochs), val_acc, label ="validation accuracy")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()


def training(model:SwinResModel, 
             train_dataloader:DataLoader,
             val_dataloader:DataLoader, 
             showVisualization:bool,
             criterion, optimizer, epochs, device):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(epochs):
        model.train()
        t_loss = 0
        t_acc = 0
        for i, data in enumerate(tqdm(train_dataloader)):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            y_hat = model(imgs)
            loss = criterion(y_hat, labels)
            loss.backward()
            optimizer.step()
            
            t_loss+=loss.item()
            prediction_indices = torch.argmax(y_hat,1)
            correct = 0
            correct += (prediction_indices == labels).sum().item()
            t_acc += correct/labels.size(0)

        
        train_loss.append(t_loss/len(train_dataloader))
        train_acc.append(t_acc/len(train_dataloader))

        model.eval()
        v_loss = 0
        v_acc = 0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                imgs, labels = data
                imgs = imgs.to(device)
                labels = labels.to(device)
                y_hat = model(imgs)
                loss = criterion(y_hat, labels)
                v_loss+=loss.item()

                prediction_indices = torch.argmax(y_hat, 1)
                correct = 0
                correct += (prediction_indices==labels).sum().item()
                v_acc += correct/labels.size(0)
               
    
        val_loss.append(v_loss/len(val_dataloader))
        val_acc.append(v_acc/len(val_dataloader)) 

        print(f'Epoch {epoch+1}  Train Loss: {train_loss[epoch]:.2f},  Train accuracy: {train_acc[epoch]:.2f}, Validation Loss: {val_loss[epoch]:.2f},  Validation accuracy: {val_acc[epoch]:.2f}')
    
    torch.save(model.state_dict(), '.checkpoint/swinres.pt')

    if(showVisualization):
        visualizeResult(train_loss, train_acc, val_loss, val_acc, epochs)




if __name__ == '__main__' :

    df = pd.read_csv('./dataset/raw/aptos-eye/train.csv')
    train_df, val_df = train_test_split(df,test_size=0.15,random_state=8)
    
    transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 40
    epochs = 30

    train_dataset = CustomDataset(df = train_df, img_dir_path="./dataset/raw/aptos-eye/train_images", transform=transform)
    val_dataset = CustomDataset(df = val_df, img_dir_path="./dataset/raw/aptos-eye/train_images", transform=transform)

    train_dataloader = DataLoader (dataset= train_dataset, batch_size=batch_size, shuffle = True)
    val_dataloader = DataLoader (dataset= val_dataset, batch_size=batch_size, shuffle = True)

    model = SwinResModel().to(device)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)


    training(model,train_dataloader, val_dataloader,True, criterion, optimizer, epochs, device)



