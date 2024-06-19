import logging
import os
import torch
import torch.nn as nn
import numpy as np
from MyDataset import LSTMDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Load Dataset
def LoadDataset(BATCH_SIZE = 1, seq_len = 270, input_size = 1000):
    frames=60
    for i in range(21):  # 21 combinations
        x = torch.load(f'LSTMTData/LSTMTrainX_{i}_{input_size}_{frames}')
        y = torch.load(f'LSTMTData/LSTMTrainY_{i}_{input_size}_{frames}')
        c = torch.ones_like(y)*i
        if i == 0:
            X = x
            Y = y
            C = c
        else:
            X = torch.cat((X, x), dim=0)
            Y = torch.cat((Y, y), dim=0)
            C = torch.cat((C, c), dim=0)

    X = X.detach()  
    Y = Y.detach()
    # Difference sequence length:
    X = X[:, 0:seq_len, :]
    print('X.shape :', X.shape)

    full_train_dataset = LSTMDataset(X, Y, C)

    train_data_loader = DataLoader(dataset = full_train_dataset,
                                batch_size = BATCH_SIZE,
                                shuffle = True,
                                num_workers=2)

    for i in range(21):  # 21 combinations
        x = torch.load(f'LSTMTrainingData/LSTMTestX_{i}_{input_size}')
        y = torch.load(f'LSTMTrainingData/LSTMTestY_{i}_{input_size}')
        c = torch.ones_like(y)*i
        if i == 0:
            X = x
            Y = y
            C = c
        else:
            X = torch.cat((X, x), dim=0)  
            Y = torch.cat((Y, y), dim=0) 
            C = torch.cat((C, c), dim=0)  

    X = X.detach() 
    Y = Y.detach()

    # Difference sequence length:
    X = X[:, 0:seq_len, :]
    print('X.shape :', X.shape)

    full_test_dataset = LSTMDataset(X, Y, C)
    eval_data_loader = DataLoader(dataset = full_test_dataset,
                                batch_size = BATCH_SIZE,
                                shuffle = True,
                                num_workers = 2)

    return train_data_loader, eval_data_loader

# Train model
def train(train_data_loader, eval_data_loader, model, criterion, optimizer, num_epoch, input_size, log_step_interval, save_step_interval, eval_step_interval, save_path, resume="", seq_len=270,lr=0.1,ep=20):
    start_epoch = 0
    start_step = 0
    max_eval_acc = 0.
    if resume !="":
        logging.warning(f"loading from {resume}")
        checkpoint = torch.load(resume, map_location=torch.device("cuda")) 
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']

    model.cuda() # model copy

    for epoch_index in range(start_epoch, num_epoch):
        num_batches = len(train_data_loader)
        for batch_index, (data, labels, c_labels) in enumerate(train_data_loader):
            optimizer.zero_grad()
            step = num_batches*(epoch_index)+batch_index+ 1 + start_step
            labels = labels.cuda()
            data = data.cuda()
            target = model(data)
            labels = labels.type(torch.LongTensor).cuda()
            loss = criterion(target, labels)
            loss.backward()
            optimizer.step()

            if step % log_step_interval == 0:
                logging.warning(f"epoch_index: {epoch_index}, batch_index: {batch_index}, loss: {loss }")

            if step % eval_step_interval == 0:
                logging.warning("start to do evalution...")
                model.eval()
                eval_loss= 0.
                total_acc_account = 0.
                total_account = 0.
                M = 0.
                eval_f1 = 0.
                matrix = torch.zeros(21,3)
                times = 0

                for eval_batch_index, (eval_data, eval_labels, c_labels) in enumerate(eval_data_loader):
                    total_account += eval_data.shape[0]

                    eval_data = eval_data.cuda()
                    eval_labels = eval_labels.cuda() 

                    outputs = model(eval_data) 
                    eval_labels = eval_labels.type(torch.LongTensor).cuda()

                    eval_loss = criterion(outputs, eval_labels)

                    # Accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct = (predicted == eval_labels).sum().item()
                    total_acc_account += correct

                    # mistakes
                    predicted = predicted.cpu()
                    eval_labels = eval_labels.cpu()

                    ind = (np.where(predicted != eval_labels))
                    M += len(ind[0])
                    for i in ind[0]:
                        row = c_labels[i].item()
                        row = int(row)
                        true_value = int(eval_labels[i].item())
                        pred_value = int(predicted[i].item())
                        if (true_value == 0 and pred_value == 1):
                            matrix[row, 0] += 1  # if true value is 0 and predicts 1
                        elif (true_value == 1 and pred_value == 0):
                            matrix[row, 1] += 1
                        else:
                            matrix[row, 2] += 1

                    eval_f1 += f1_score(eval_labels, predicted, average='micro')
                    times += 1
                eval_f1 = eval_f1/times

                logging.warning(f"eval_loss: {eval_loss}, eval_acc: {total_acc_account / total_account}, max_eval_acc:{max_eval_acc}, eval_f1:{eval_f1},seq_len:{seq_len},lr:{learning_rate}")
                model.train()

                if total_acc_account / total_account > max_eval_acc:
                    torch.save(model, f'model_{input_size}_{seq_len}_{lr}_{ep}')
                    max_eval_acc = total_acc_account / total_account


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=500, num_layers=6, num_classes=3):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.Linear(256,num_classes)
        )

    def forward(self, x): 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        out, (hidden, cell) = self.lstm(x, (h0, c0))
        out =  hidden[-1, :, :]
        preds = self.fc(out)

        return preds

if __name__ == "__main__":
    if torch.cuda.is_available():
        logging.warning("Cuda is available!")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIVLE_DEVICES"] = "0" 
    else:
        logging.warning("Cuda is not availble! Exit!")


    # parameter
    learning_rate = 0.00001
    num_epochs = 500
    input_size = 1000  
    BATCH_SIZE = 1
    seq_len = 270

    train_data_loader, eval_data_loader = LoadDataset(BATCH_SIZE=BATCH_SIZE, seq_len=seq_len, input_size=input_size)

    # LSTM model
    model = LSTM(input_size, hidden_size=1000, num_layers=2, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    train(train_data_loader=train_data_loader, eval_data_loader=eval_data_loader,
            model=model, criterion=criterion, optimizer=optimizer,
            num_epoch=num_epochs,input_size=input_size,
            log_step_interval=500, save_step_interval=500, eval_step_interval=500,
            save_path='./LSTMTrainingPath/',
            resume="",seq_len=seq_len,lr=learning_rate,ep=num_epochs)
