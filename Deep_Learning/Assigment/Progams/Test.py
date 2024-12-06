import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
from Utils.GetData import data
from torch import optim
from Models.cnn import CNN
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def main():
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    folds=[1, 2, 3, 4, 5]
    ori_path = './Original Images/Original Images/FOLDS'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for fold in folds:
        test_loader = DataLoader(data(ori=ori_path, folds=[fold], subdir='Valid'), batch_size=BATCH_SIZE, shuffle=True)
    num_classes = 6
    model = CNN(input_dim=128, input_c=3, output=num_classes, hidden_dim=128, dropout=0.5)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(),lr=LEARNING_RATE, weight_decay=1e-4)
    
    checkpoint = torch.load('Models_10.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss_batch = checkpoint['loss']
    
    prediction, ground_truth = [], []
    with torch.no_grad():
        model.eval()
        for batch, (src, trg) in enumerate(test_loader):
            src, trg = src.to(device), trg.to(device)
            src = torch.permute(src, (0, 3, 1, 2))

            pred = model(src)
            prediction.extend(torch.argmax(pred,dim=1).detach().cpu().numpy())
            ground_truth.extend(torch.argmax(trg, dim=1).detach().cpu().numpy())
    
    classes = ['Chickenpox', 'Cowpox', 'Healthly', 'HFMD', 'Measles', 'Monkeypox']
    
    # Build confusion matrix
    cf_matrix = confusion_matrix(ground_truth, prediction)
    print(cf_matrix)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')
    
    print(f"accuracy score =  {(accuracy_score(ground_truth, prediction))*100:.2f}%")
    print(f"precision score = {(precision_score(ground_truth, prediction, average='weighted'))*100:.2f}%")
    print(f"recall score = {(recall_score(ground_truth, prediction, average='weighted'))*100:.2f}%")
    print(f"f1 score score = {(f1_score(ground_truth, prediction, average='weighted'))*100:.2f}%")


if __name__ == "__main__":
    main()