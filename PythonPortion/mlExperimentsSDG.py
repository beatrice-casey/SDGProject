import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import torch
import time
import psutil

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Palatino'], 'size': 24})
rc('text', usetex=False)
import matplotlib


def main():
    # get labels
    data = pd.read_csv("./vulns.csv")
    # get embeddings
    x = pd.read_csv("./graph2vec/features/graphEmbedding_sdg.csv")

    data["vulnerable"] = data["vulnerable"].astype(int)

    y = data.iloc[:, 2]
    x['type'] = x['type'].str.replace('sdg_', '')
    #print(x)
    y_label = y.iloc[x['type']]
    x = x.drop(columns="type")

    dataset = x.join(y_label)

    y = dataset['vulnerable']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25,
                                                      random_state=1)  # 0.25 x 0.8 = 0.2

    x_train_tensors = torch.FloatTensor(x_train.values)
    x_test_tensors = torch.FloatTensor(x_test.values)
    x_val_tensors = torch.FloatTensor(x_val.values)

    y_train_tensors = torch.tensor(y_train.values)
    y_test_tensors = torch.tensor(y_test.values)
    y_val_tensors = torch.tensor(y_val.values)

    import torch.nn as nn
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.layers(x)
            return x

    model = MLP()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    criterion = torch.nn.BCELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.eval()
    y_pred = model(x_test_tensors)
    before_train = criterion(y_pred.squeeze(), y_test_tensors.float())
    print('Test loss before training', before_train.item())

    def train():
        model.train()

        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_train_tensors)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train_tensors.float())
        # Backward pass
        loss.backward()
        optimizer.step()

        return loss

    def evaluate(x_tensors, y_tensors):
        model.eval()
        y_pred = model(x_tensors)
        loss = criterion(y_pred.squeeze(), y_tensors.float())
        return accuracy_score(y_tensors.detach().cpu().numpy(), np.rint(y_pred.cpu().detach().numpy())), loss

    import time
    import psutil
    print("Starting training...")
    train_losses = []
    val_losses = []
    val_acc_list = []
    train_acc_list = []
    mean_train_losses = []
    mean_val_losses = []
    best_loss = 1000
    early_stopping_counter = 0
    start_time = time.time()
    for epoch in range(200):
        if early_stopping_counter <= 5:
            loss = train()
            train_losses.append(loss)
            train_acc, train_loss = evaluate(x_train_tensors, y_train_tensors)
            val_acc, val_loss = evaluate(x_val_tensors, y_val_tensors)
            val_losses.append(val_loss)
            val_acc_list.append(val_acc)
            train_acc_list.append(train_acc)
            # mean_train_losses.append(np.mean(train_losses))
            # mean_val_losses.append(np.mean(val_losses))
            if float(val_loss) < best_loss:
                best_loss = val_loss
                # Save the currently best model
                early_stopping_counter = 0
            else:
                early_stopping_counter = 1
            print(
                f"Epoch {epoch} | Train Loss {loss} | Train Accuracy{train_acc} | Validation Accuracy{val_acc} | Validation loss{best_loss}")

        else:
            print("Early stopping due to no improvement.")
            break

    end_time = time.time()
    total_time = end_time - start_time
    print("total time taken for model training:", total_time, "seconds")
    print(f"Finishing training with best validation loss: {best_loss}")
    psutil.getloadavg()

    with torch.no_grad():
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 13))
        ax1.plot(train_losses, label="$train\_loss$")
        ax1.plot(val_losses, label="$val\_loss$")
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epochs')
        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend(lines, labels, loc='best')

        ax2.plot(train_acc_list, label='train\_acc')
        ax2.plot(val_acc_list, label='val\_acc')
        ax2.legend()
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epochs')
        # plt.rc('font', family='serif)
        plt.savefig('SDG.png')

    # Analyze the results
    test_batch = next(iter(x_test_tensors))
    with torch.no_grad():
        y_pred = model(x_test_tensors)
        df = pd.DataFrame()
        df["y_pred"] = y_pred.tolist()
    df

    # y_test = y_test_tensors.detach().cpu().numpy()
    y_pred = np.rint(y_pred)
    # confusion_matrix(y_test, y_pred)
    # plot_confusion_matrix(model, x_test_tensors, y_test_tensors)
    # plt.show()
    accuracy = accuracy_score(y_test_tensors, y_pred)
    precision = precision_score(y_test_tensors, y_pred, zero_division=1)
    recall = recall_score(y_test_tensors, y_pred, zero_division=1)
    F1 = f1_score(y_test_tensors, y_pred, zero_division=1)
    print(" Accuracy : {0} \n Precision : {1} \n Recall : {2} \n F1 : {3}".format(accuracy, precision, recall, F1))

    model.eval()
    y_pred = model(x_test_tensors)
    after_train = criterion(y_pred.squeeze(), y_test_tensors.float())
    print('Test loss after training', after_train.item())



main()
