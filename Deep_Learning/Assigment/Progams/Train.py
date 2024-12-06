import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch import optim
from Utils.GetData import data
from  Models.cnn import CNN
import matplotlib.pyplot as plt

def main():
    folds=[1, 2, 3, 4, 5]
    ori_path = './Original Images/Original Images/FOLDS'
    aug_path = './Augmented Images/Augmented Images/FOLDS_AUG'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 16
    EPOCH = 100
    LEARNING_RATE = 0.001
    
    for fold in folds:
        train_loader = DataLoader(ConcatDataset([data(ori=ori_path, folds=[fold], subdir='Train'), 
                                                 data(aug=aug_path, folds=[fold], subdir='Train')]), batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(data(ori=ori_path, folds=[fold], subdir='Valid'), batch_size=BATCH_SIZE, shuffle=False)

    num_classes = 6
    model = CNN(input_dim=224, input_c=3, hidden_dim=128, output=num_classes, dropout=0.5)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(),lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, patience=10)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    loss_train_all = []
    loss_valid_all = []
    for epoch in range(EPOCH):
        loss_train = 0
        loss_valid = 0
        total = 0
        correct = 0
        model.train()
        for batch, (src, trg) in enumerate(train_loader):
            src, trg = src.to(device), trg.to(device)
            src = torch.permute(src, dims=(0, 3, 1, 2))
            pred = model(src).to(device)
            trg = trg.argmax(dim=1)
            loss = criterion(pred, trg)
            loss_train += loss.cpu().detach().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()       

        model.eval()
        for batch, (src, trg) in enumerate(valid_loader):
            src, trg = src.to(device), trg.to(device)
            src = torch.permute(src, dims=(0, 3, 1, 2))
            pred = model(src).to(device)
            trg = trg.argmax(dim=1)
            loss = criterion(pred, trg)
            loss_valid += loss.cpu().detach().numpy()
            _, predicted = torch.max(pred, 1) 
            total += trg.size(0)               
            correct += (predicted == trg).sum().item()
        
        accuracy = (correct / total)*100
        print(f"Epoch: {epoch + 1}, Train Loss: {loss_train / len(train_loader):.4f}, "
          f"Valid Loss: {loss_valid / len(valid_loader):.4f}, Accuracy: {accuracy:.2f}%")
        
        scheduler.step(loss_valid)
        loss_train_all.append(loss_train/(len(train_loader)))
        loss_valid_all.append(loss_valid/len(valid_loader))
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_train / (len(train_loader)),
            }, f'Models_{epoch+1}.pt')
            
    plt.plot(range(EPOCH), loss_train_all, color="#931a00", label='Training')
    plt.plot(range(EPOCH), loss_valid_all, color="#3399e6", label='Testing')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./Hasil/training.png")

if __name__=="__main__":
    main()