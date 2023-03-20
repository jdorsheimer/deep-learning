# multi-class model
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5, # gives the clusters higher variance
                            random_state=RANDOM_SEED)

# turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# split data into train and test
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

# plot data
plt.figure(figsize=(10,7))
plt.scatter(x=X_blob[:,0],
            y=X_blob[:,1],
            c=y_blob,
            cmap=plt.cm.RdYlBu)

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# build model
class BlobModel(nn.Module):
  def __init__(self, input_features, output_features, hidden_units=8):
    '''
    Initializes multi-class classification model.
    Args:
      input_features (int): Number of input features to the model.
      output_features (int): Number of output features to the model (number of output classes).
      hidden_units (int): Number of hidden units between layers, default 8.
    '''
    super().__init__()
    self.linear_layer_stack = nn.Sequential(
      nn.Linear(in_features=input_features, out_features=hidden_units),
      nn.ReLU(),
      nn.Linear(in_features=hidden_units, out_features=hidden_units),
      nn.ReLU(),
      nn.Linear(in_features=hidden_units, out_features=output_features)
    )
  def forward(self, x):
    return self.linear_layer_stack(x)

# create an instance of BlobModel and send to target device
model_4 = BlobModel(input_features=2,
                    output_features=4,
                    hidden_units=8).to(device)

# loss function - measurs how incorrect model's prediction is
loss_fn = nn.CrossEntropyLoss()

# optimizer - updated model parameters to reduce the loss
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1)

def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_pred))*100
  return acc

# raw outputs of model - logits
model_4.eval()
with torch.inference_mode():
  y_logits = model_4(X_blob_test.to(device))
# print(y_logits[:5])

# convert logits (raw output of model) > pred probs (use torch.softmax) > pred labels (take the argmax of the pre probs)
# converts logits to pred probs
y_pred_probs = torch.softmax(y_logits,dim=1)
# print(y_pred_probs[:5])

# convert pred probs to pred labels
y_pred_labels = torch.argmax(y_pred_probs,dim=1)
# print(y_pred_labels[:5])
# print(y_blob_test[:5])

# training loop
# fit multi-class model to the data
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#set number of epochs
epochs = 100

# put data to target device
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)


# loop through data
for epoch in range(epochs):
  model_4.train()

  y_logits = model_4(X_blob_train)
  

  
  
  y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
  loss = loss_fn(y_logits, y_blob_train)
  acc = accuracy_fn(y_true=y_blob_train,
                    y_pred=y_pred)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # testing code
  model_4.eval()
  with torch.inference_mode():
    test_logits = model_4(X_blob_test)
    test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

    test_loss = loss_fn(test_logits, y_blob_test)
    test_acc = accuracy_fn(y_true=y_blob_test,
                           y_pred=test_preds)
    
  if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Loss {loss:.4f} | Acc: {acc:.2f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")


# make and evaluate predictions with multi-class model
model_4.eval()
with torch.inference_mode():
  y_logits = model_4(X_blob_test)

# view the first 10 predictions

print(y_logits[:10])
y_pred_probs = torch.softmax(y_logits,dim=1)
print(y_pred_probs[:10])
y_pred_labels = torch.argmax(y_pred_probs, dim=1)
print(y_pred_labels[:10])
print(y_blob_test[:10])
