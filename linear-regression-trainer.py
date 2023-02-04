import torch
from torch import nn
import matplotlib.pyplot as plt

# check version
torch.__version__

# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# create data
weight = 0.6
bias = 0.2

# create range values
start = 0
stop = 1
step = 0.02

# create X and y (features and labels)
X = torch.arange(start,stop,step).unsqueeze(dim=1)
y = weight * X + bias

# split data
train_split = int(0.8*len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

def plot_predictions(train_data=X_train, train_labels=y_train,test_data=X_test,test_labels=y_test,predictions=None):
  """
  plots training data, test data, and compares predictions
  """
  plt.figure(figsize=(10,7))

  # plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

  # plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
  if predictions is not None:
    #plot predictions if they exist
    plt.scatter(test_data,predictions,c="r",s=4,label="Predictions")
  
  plt.legend(prop={"size": 14})

# plot the data
plot_predictions(X_train, y_train, X_test, y_test)

# build a torch model
class LinearRegressionModelV2(nn.Module):
  def __init__(self):
    super().__init__()
    # use nn.Linear() to create model parameters
    self.linear_layer = nn.Linear(in_features=1,
                                  out_features=1)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear_layer(x)

# set manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()

# set model to use target device
model_1.to(device)
# next(model_1.parameters()).device

# loss function
loss_fn = nn.L1Loss() # same as MAE

# optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01) #Adam() is another model

# training loop
torch.manual_seed(42)

epochs = 150

# device-agnostic code

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# capture loss to test efficacy
epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
  model_1.train()

  # forward pass
  y_pred = model_1(X_train)

  # calculate loss
  loss = loss_fn(y_pred,y_train)

  # optimizer zero grad
  optimizer.zero_grad()

  # back-propagation
  loss.backward()

  # optimizer
  optimizer.step()

  # testing
  model_1.eval()
  with torch.inference_mode():
    test_pred = model_1(X_test)
    test_loss = loss_fn(test_pred, y_test)
  
  epoch_count.append(epoch)
  loss_values.append(loss)
  test_loss_values.append(test_loss)

  if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")
