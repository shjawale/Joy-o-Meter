#!/usr/bin/env python
# coding: utf-8

# # CS224 - Spring 2024 - HW1 - Joy-o-Meter

import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Load dataset and visualize
train_file = 'EI-reg-En-joy-train.txt'
val_file = '2018-EI-reg-En-joy-dev.txt'
df_train = pd.read_csv(train_file, sep='\t')
df_val = pd.read_csv(val_file, sep='\t')

tweets_train = df_train['Tweet'].tolist()  # Create a list of tweets
tweets_val = df_val['Tweet'].tolist()

# Create a list of intensity scores
y_train = torch.tensor(df_train['Intensity Score'], dtype=torch.float32)  # match to dtype of embedding
y_val = torch.tensor(df_val['Intensity Score'], dtype=torch.float32)

print('Score - Tweet')
for i in range(5):
    print('{:0.2f} - {}'.format(y_train[i], tweets_train[i]))

model_name="bert-base-uncased"  # Many possibilities on huggingface.com

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load pre-trained model
model = AutoModel.from_pretrained(model_name)

sentence = "Hello, this is an example sentence!"

def embed_sentence(model, tokenizer, sentence):
    """Function to embed a sentence as a vector using a pre-trained model."""
    inputs = tokenizer(sentence, return_tensors="pt")  # Turn text into one-hot vectors

    # Forward pass, get hidden states
    with torch.no_grad():
        outputs = model(**inputs)

    # For BERT, last hidden state is the embedding of each item in the sequence
    return outputs.last_hidden_state[0].mean(dim=0)  # use mean embedding, for hw
    # return outputs.last_hidden_state[0, 0]  # use CLS embedding, another good choice


train_list = [embed_sentence(model, tokenizer, i) for i in tweets_train]
X_train = torch.stack(train_list, 0)
print(X_train.shape)

val_list = [embed_sentence(model, tokenizer, i) for i in tweets_val]
X_val = torch.stack(val_list, 0)
print(X_val.shape)


# ## Define the model
class MyLinearNet(torch.nn.Module):
    def __init__(self, embedding_size):
        super().__init__()        # init superclass - enables many pytorch model capabilities
        self.d = embedding_size   # Often convenient to store this (not a "Parameter" though as we don't train it)
        self.weights = torch.nn.Parameter(torch.randn(d))
        self.bias = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, x):#when you run model, this is what runs
        """Implement a linear model"""
        # It should work on a single x, or a batch
        xw = x @ self.weights
        y_hat = torch.add(xw, self.bias)
        return y_hat

    def fit(self, X, y):
        """Given a data matrix, X, and a vector of labels for each row, y,
        analytically fit the parameters of the linear model."""

        # (a) First, construct the augmented data matrix as discussed in class
        col_of_ones = torch.ones(len(X))
        aug = torch.cat((X,col_of_ones.unsqueeze(1)),1)
        print(col_of_ones.shape)

        # (b) Next, use matrix multiplication and torch.linalg.inv to implement the analytic solution
        M = aug.T @ aug             #XT X
        Minv = torch.linalg.pinv(M)  #(XTX)^-1
        print(aug.T.shape)
        print(y.shape)
        print(aug.shape)
        XTy = aug.T @ y             #XT y
        w = Minv @ XTy              #(XTX)^-1 XTy
        #print(w)

        # (c) Put the solution (which includes weights and biases) into parameter
        # Use "data" to update parameters without affecting computation graph
        # (Kind of a subtle point - no need to modify my code below)
        self.weights.data = w[:self.d]
        self.bias.data = w[-1]


col_of_ones = torch.ones(len(X_train))
aug = torch.cat((X_train,col_of_ones.unsqueeze(1)),1)
print(aug.shape)
print(y_train.shape)

print(len(X_train))


# ## Results
def loss(model, X, y):
  y_hat = model(X)
  #print(X.shape)
  sum_y_diff = 0
  y_diff = y - y_hat
  y_diff_sq = y_diff **2
  return torch.mean(y_diff_sq)

d = X_train.shape[1]  # embedding dimension
mymodel = MyLinearNet(d)
mymodel.forward(X_train)

print()
print()
loss_train = loss(mymodel, X_train, y_train)
loss_val = loss(mymodel, X_val, y_val)
print("\nLoss on train and validation BEFORE fitting.\nTrain: {:0.3f}, Val: {:0.3f}".format(loss_train, loss_val))

print()
print()
mymodel.fit(X_train, y_train)

loss_train = loss(mymodel, X_train, y_train)
loss_val = loss(mymodel, X_val, y_val)

print("\nLoss on train and validation AFTER fitting.\nTrain: {:0.3f}, Val: {:0.3f}".format(loss_train, loss_val))


# Create a scatter plot of the actual vs. predicted values of `y` using this function.
def plot(y_train, y_hat_train, y_val, y_hat_val):
    fig, ax = plt.subplots(1)
    ax.scatter(y_train, y_hat_train, alpha=0.4, label='train')
    ax.scatter(y_val, y_hat_val, label='val')
    ax.set_xlabel('y - ground truth joy intensity')
    ax.set_ylabel('Predicted y')
    ax.legend()


with torch.no_grad():  # remember to turn off auto gradient tracking
    y_hat_train = mymodel(X_train)
    y_hat_val = mymodel(X_val)

plot(y_train, y_hat_train, y_val, y_hat_val)


# Put in a sample sentence of your own construction and output the "joy meter" for a happy and sad sentence
happy = "I am never happier than when I'm doing homework :) LOL"
sad = "I am so tired of homework right now"

happy_embedding = embed_sentence(model, tokenizer, happy)
y_hat_happy = mymodel(happy_embedding)

sad_embedding = embed_sentence(model, tokenizer, sad)
y_hat_sad = mymodel(sad_embedding)

print('{:0.2f} - {}'.format(y_hat_happy,happy))
print('{:0.2f} - {}'.format(y_hat_sad, sad))


# ##Works Cited
# *   https://stackoverflow.com/questions/65804689/with-bert-text-classification-valueerror-too-many-dimensions-str-error-occur
