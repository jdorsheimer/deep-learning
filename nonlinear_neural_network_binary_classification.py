# make and plot data

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

n_samples = 1000

X, y = make_circles(n_samples,
                    noise =0.03,
                    random_state=42)
plt.scatter(x=X[:,0],
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu)

import torch
import numpy as np
from sklearn.model_selection import train_test_split

# turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# split into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# X_train[:5],y_train[:5]

# nn.ReLU = rectified linear unit function, element-wise - turns negative inputs to zero, positive inputs linearly upscaled
# build a model with nonlinearity
from torch import nn
class CircleModelV2(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(in_features=2,out_features=10) 
    self.layer_2 = nn.Linear(in_features=10,out_features=10) 
    self.layer_3 = nn.Linear(in_features=10,out_features=1)
    self.relu = nn.ReLU() # nonlinear activation function
    self.sigmoid = nn.Sigmoid # can be used here, but is better to be embedded in logits
  def forward(self, x):
    return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_pred))*100
  return acc

device = "cuda" if torch.cuda.is_available() else "cpu"

model_3 = CircleModelV2().to(device)

# loss function - binary cross entropy
loss_fn = nn.BCEWithLogitsLoss()

# optimizer - stochastic gradient descent
optimizer = torch.optim.SGD(params=model_3.parameters(),lr=0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# set up epochs

epochs = 10000

# put data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
  # training
  model_3.train()
  #forward pass
  y_logits = model_3(X_train).squeeze()
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
  model_3.eval()
  with torch.inference_mode():
    #forward pass
    test_logits = model_3(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    #calculate loss
    test_loss = loss_fn(test_logits,y_test)
    test_acc = accuracy_fn(y_true=y_test,
                           y_pred=test_pred)
  # print results
  if epoch % 1000 == 0:
      print(f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.2f}% | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}")

from helper_functions import plot_predictions, plot_decision_boundary

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_3, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test)
