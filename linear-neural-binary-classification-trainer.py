import sklearn
from sklearn.datasets import make_circles

# make samples

n_samples = 1000

# create circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

import pandas as pd

circles = pd.DataFrame({"X1": X[:,0],
                        "X2": X[:,1],
                        "label": y})

# circles.head(10)

import matplotlib.pyplot as plt

plt.scatter(x=X[:,0],
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu)

# X_sample = X[0]
# y_sample = y[0]

# X_sample,y_sample

import torch
from torch import nn
import numpy as np

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# split data into training and test sets
from sklearn.model_selection import train_test_split

# training first, labels second
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, # 20% of data will be test and 80% will be train
                                                    random_state=42)

# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# construct a model that subclasses nn.Module
class CircleModelV0(nn.Module):
  def __init__(self):
    super().__init__()
    # create 2 nn.Linear layers capable of handling data shapes
    self.layer_1 = nn.Linear(in_features=2,
                             out_features=5)
    self.layer_2 = nn.Linear(in_features=5,
                             out_features=1)
  def forward(self, x):
    return self.layer_2(self.layer_1(x)) # x -> layer_1 -> layer_2

# instantiate instance of model class and pass to the target device
model_0 = CircleModelV0().to(device)
# print(model_0.state_dict())
# replicate model with nn.Sequential
# model_0 = nn.Sequential(
#     nn.Linear(in_features=2,out_features=5),
#     nn.Linear(in_features=5,out_features=1)
# )

# layer_n weight = out_features x in_features; l1= 5 x 2, l2=1 x 5
# make predictions
with torch.inference_mode():
  untrained_preds = model_0(X_test.to(device))

# X_test[:10], y_test[:10]

# loss function
# for linear regression, use MAE (mean absolute error) or MSE (mean square error) 
# for classification, use BCE (binary cross entropy) or categorical cross entropy (CCE, also shortened to cross entropy)

# softmax - multi-classification
# sigmoid -  binary classification
#BCELoss = requires inputs to have gone through sigmoid activation

# the below is equivalent to nn.Sequential(nn.Sigmoid(), nn.BCELoss()) - combines both, but the below is more numerically stable

loss_fn = nn.BCEWithLogitsLoss() # sigmoid activation function included with BCE

# optimizer
# SGD and Adam, but torch has many other options, see torch.optim
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.1) # stochatisc gradient descent

# accuracy - out of N examples, what % does the model get correct?
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_pred))*100
  return acc

# create logit function > labels functions
# raw logits -> prediction probabilities -> prediction labels
# pass logits to sigmoid (binary classification) or softmax (multiclass classification)
# convert prediction probabilities to prediction labels by rounding or using argmax()
model_0.eval()
with torch.inference_mode():
  y_logits = model_0(X_test.to(device))[:5]
# y_logits

# sigmoid activation function on logits to turn them into prediction probabilities
y_pred_probs = torch.sigmoid(y_logits)

# find the predicted labels
y_preds = torch.round(y_pred_probs)

# in full (logits > pred_probs > pred_labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# check for equality
# print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# remove extra dimension
y_preds.squeeze()

# train model
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 100

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
  model_0.train()

  # forward pass
  y_logits = model_0(X_train).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits)) # logits > pred probs > pred labels

  # calculate loss + accuracy
  loss = loss_fn(y_logits, # nn.BCEWithLogitsLoss expects raw logits as input
                 y_train) # predictions come first, then data
  acc = accuracy_fn(y_true=y_train,
              y_pred=y_pred)


  # optimizer zero grad
  optimizer.zero_grad()

  # loss backward (back propagation)
  loss.backward()

  # optimizer step
  optimizer.step()

  # testing
  model_0.eval()
  with torch.inference_mode():
    # forward pass
    test_logits = model_0(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))

    # calculate test loss/accuracy
    test_loss = loss_fn(test_logits,
                        y_test)
    test_acc = accuracy_fn(y_true=y_test,
                           y_pred=test_pred)
    
  # print out results
  # if epoch % 10 == 0:
  #     print(f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.2f}% | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}")

# visualize predictions
# import requests
# from pathlib import Path

# # download helper functions from external repo if not already downloaded
# if not Path("helper_functions.py").is_file():
#   request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
#   with open("helper_functions.py", "wb") as f:
#     f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

# plot decision boundary of the model

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)

# to improve a model, only make 1 change at a time
# improve model by: add more layers, add more hidden units/parameters, train for longer (epochs++), change activation functions (sigmoid > ?), learning rate (lr = lower), change loss function (BCE > ?)
# imrove data by: 

class CircleModelV1(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(in_features=2,out_features=10)
    self.layer_2 = nn.Linear(in_features=10, out_features=10)
    self.layer_3 = nn.Linear(in_features=10, out_features=1)
  def forward(self, x):
    # z = self.layer_1(x)
    # z = self.layer_2(z)
    # z = self.layer_3(z)
    return self.layer_3(self.layer_2(self.layer_1(x))) # faster because does not assign multiple variables

# set new model
model_1 = CircleModelV1().to(device)

# loss function
loss_fn = nn.BCEWithLogitsLoss()

# optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),lr=0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# set up epochs

epochs = 1000

# put data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
  # training
  model_1.train()
  #forward pass
  y_logits = model_1(X_train).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits)) # logits > pred probs > pred labels
  # calculate loss/accuracy
  loss = loss_fn(y_logits, y_train)
  acc = accuracy_fn(y_true=y_train,
                    y_pred=y_pred)
  # optimizer zero grad
  optimizer.zero_grad()
  # loss backwards (back prop)
  loss.backward()
  #optimizer step (grad descent)
  optimizer.step()
  #testing
  model_1.eval()
  with torch.inference_mode():
    #forward pass
    test_logits = model_1(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    #calculate loss
    test_loss = loss_fn(test_logits,y_test)
    test_acc = accuracy_fn(y_true=y_test,
                           y_pred=test_pred)
  # print results
  # if epoch % 100 == 0:
  #     print(f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.2f}% | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
