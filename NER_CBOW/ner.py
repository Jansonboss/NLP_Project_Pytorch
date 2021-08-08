import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


# if a word is not in your vocabulary use len(vocabulary) as the encoding
class NERDataset(Dataset):
    def __init__(self, df_enc):
        self.df = df_enc
        # give a sliding window of fixed length=5  
        self.x = [df_enc.iloc[i:5+i].word.values for i in range(len(df_enc)) if i+5<=len(df_enc)]
        self.y = [df_enc.iloc[2+i].label for i in range(len(df_enc)) if i+5<=len(df_enc)]

    def __len__(self):
        """ Length of the dataset """
        return len(self.x)

    def __getitem__(self, idx):
        """ returns x[idx], y[idx] for this dataset
        x[idx] should be a numpy array of shape (5,)
        """
        return self.x[idx], self.y[idx]


def label_encoding(cat_arr):
   """ Given a numpy array of strings returns a dictionary with label encodings.
   First take the array of unique values and sort them (as strings). 
   """
   cat_arr = list(map(str, cat_arr))
   sorted_unique_vocabs = np.sort(np.unique(cat_arr))
   vocab2index = {vocab: idx for idx, vocab in enumerate(sorted_unique_vocabs)}
   return vocab2index


def dataset_encoding(df, vocab2index, label2index):
    """Apply vocab2index to the word column and label2index to the label column
    Replace columns "word" and "label" with the corresponding encoding.
    If a word is not in the vocabulary give it the index V=(len(vocab2index))
    """
    V = len(vocab2index)
    df_enc = df.copy()
    
    vocab_index = [vocab2index.get(vocab, V) for vocab in df["word"].values]
    label_index = [label2index.get(label, V) for label in df["label"].values]
    df_enc["word"] = vocab_index
    df_enc["label"] = label_index
    return df_enc


class NERModel(nn.Module):
    def __init__(self, vocab_size, n_class, emb_size=50, seed=3):
        """Initialize an embedding layer and a linear layer
        """
        super(NERModel, self).__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(5*emb_size, n_class)
        self.dropout = nn.Dropout(p=0.6)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        """Apply the model to x
        1. x is a (N,5). Lookup embeddings for x
        2. reshape the embeddings (or concatenate) such that x is N, 5*emb_size => .flatten works
        3. Apply a linear layer
        """
        x = self.emb(torch.LongTensor(x)).flatten(start_dim=1)
        x = self.linear(x)
        # x = self.softmax(x)
        return x

def get_optimizer(model, lr = 0.01, wd = 0.0):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optim

def train_model(model, optimizer, train_dl, valid_dl, epochs=10):
    # training the model
    model.train()
    for i in range(epochs):
        sum_loss, total_batch = 0, 0
        for x, y in train_dl:
            batch_size = x.shape[0]
            # forawd + backward + optimize
            optimizer.zero_grad()
            y_hat = model(x)
            loss = F.cross_entropy(input=y_hat, target=y)
            loss.backward()
            optimizer.step()
            # accumulating the batch loss 
            total_batch +=  batch_size
            sum_loss += loss.item()*batch_size

        train_loss = sum_loss / total_batch
        valid_loss, valid_acc = valid_metrics(model, valid_dl)
        print("train loss  %.3f val loss %.3f and accuracy %.3f" % (train_loss, valid_loss, valid_acc))

def valid_metrics(model, valid_dl):
    # evalidate the model
    model.eval()
    correct, sum_loss, total_batch = 0, 0, 0
    for x_val, y_val in valid_dl:
        y_hat = model(x_val)
        loss = F.cross_entropy(y_hat, y_val)
        _, pred_label = torch.max(y_hat.data, 1)
        sum_loss += y_val.size(0)*loss.item()
        total_batch += y_val.size(0)
        correct += pred_label.eq(y_val.data).sum().item()

    val_loss, val_acc = sum_loss / total_batch, correct / total_batch
    return val_loss, val_acc

