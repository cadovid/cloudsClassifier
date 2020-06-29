import torch
import torchvision

import time
import pandas as pd
import sklearn.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import matplotlib.pyplot as pl
import matplotlib.image as mpimg
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

torch.cuda.empty_cache()

semeion_x = np.load("nubes_train.npy")
semeion_y = np.load("labels_train.npy")
semeion_validation_x = np.load("nubes_validation.npy")
semeion_validation_y = np.load("labels_validation.npy")
semeion_test_x = np.load("nubes_test.npy")
semeion_test_y = np.load("labels_test.npy")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Usando device = ', device)

semeion_x = torch.tensor(semeion_x, dtype=torch.float)
semeion_y = torch.tensor(semeion_y, dtype=torch.long)

semeion_validation_x = torch.tensor(semeion_validation_x, dtype=torch.float)
semeion_validation_y = torch.tensor(semeion_validation_y, dtype=torch.long)

semeion_test_x = torch.tensor(semeion_test_x, dtype=torch.float)
semeion_test_y = torch.tensor(semeion_test_y, dtype=torch.long)


class NetConv(nn.Module):
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 15, kernel_size=4)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(55815, 12)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        in_size = x.shape[0]
        if (visualiza): print('A la entrada: ', x.shape, 'Tamaño del minibatch:', in_size)
        x = F.relu(self.mp(self.conv1(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        if (visualiza): print('Flattened tensor: ', x.shape)
        x = self.fc(x)
        if (visualiza): print(x.shape)
        output = x
        if (visualiza): print(output.shape)
        return (output)


# PARÁMETROS
batchsize = 64
lr = 0.01
momentum = 0.6
seed = 1

np.random.seed(seed)
torch.manual_seed(seed)
model = NetConv().to(device)
print(model)

train_ds = TensorDataset(semeion_x, semeion_y)
validation_ds = TensorDataset(semeion_validation_x, semeion_validation_y)
test_ds = TensorDataset(semeion_test_x, semeion_test_y)
train_dl = DataLoader(train_ds, batch_size=batchsize, shuffle=True)
validation_dl = DataLoader(validation_ds, batch_size=len(validation_ds))
test_dl = DataLoader(test_ds, batch_size=len(test_ds))


for xb,yb in train_dl:
  break

xb = xb.to(device)

visualiza = True
with torch.no_grad():
    result = (model.forward(xb))

visualiza = False

print(result)

print('Tamaño a la entrada ', xb.shape)
print('Tamaño a la salida', result.shape)

# import hiddenlayer as hl
# hl.build_graph(model,xb)

torch.cuda.empty_cache()
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

list_results = []

start = time.time()
for epoch in range(0, 200):
    model = model.train()

    train_loss = train_correct = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss = criterion(output, yb)
        train_loss += loss.item()
        predicted = torch.max(output.data, dim=1)[1]
        train_correct += (predicted == yb).sum().item()
        loss.backward()
        optimizer.step()
    train_loss /= float(len(train_dl))
    train_correct /= float(len(train_ds))


    model = model.eval()
    validation_loss = validation_correct = 0
    with torch.no_grad():
        for xb, yb in validation_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)
            loss = criterion(output, yb)
            validation_loss += loss.item()
            predicted = torch.max(output.data, dim=1)[1]
            validation_correct += (predicted == yb).sum().item()
    validation_loss /= float(len(validation_dl))
    validation_correct /= float(len(validation_ds))


    model = model.eval()
    test_loss = test_correct = 0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)
            loss = criterion(output, yb)
            test_loss += loss.item()
            predicted = torch.max(output.data, dim=1)[1]
            test_correct += (predicted == yb).sum().item()
    test_loss /= float(len(test_dl))
    test_correct /= float(len(test_ds))

    result = (epoch, train_loss, train_correct, validation_loss, validation_correct,
              test_loss, test_correct)
    list_results.append(result)

    if (epoch % 10 == 0):
        print('Epoch: {} TrLoss: {} Tr: {} ValLoss: {} Val: {} TsLoss: {} Ts: {}'.format(*result[0:3], *result[3:5],
                                                                                         *result[5:]))

end = time.time()
print('tiempo', end - start)

list_res_np = np.array(list_results)
mejor_epoch_validacion = np.argmin(np.array(list_res_np)[:,3])
print('El mejor valor de la Loss en validación se obtuvo en el epoch: ',mejor_epoch_validacion)
print('Y las tasas de acierto en entrenamiento, validación y test eran:', list_res_np[mejor_epoch_validacion,[2,4,6]])

from matplotlib import pyplot as plt

plt.figure()
plt.subplots_adjust(hspace=0.5)

plt.subplot(2,1,1)
plt.plot(list_res_np[:,0], list_res_np[:,2], label="Train")
plt.plot(list_res_np[:,0], list_res_np[:,4], label="Validation")
plt.plot(list_res_np[:,0], list_res_np[:,6], label="Test")
plt.legend(loc="lower right")
plt.title('Tasa de aciertos')

plt.subplot(2,1,2)
plt.plot(list_res_np[:,0], list_res_np[:,1], label="Train")
plt.plot(list_res_np[:,0], list_res_np[:,3], label="Validation")
plt.plot(list_res_np[:,0], list_res_np[:,5], label="Test")
plt.legend(loc="upper right")
plt.title('Loss')
plt.show()

# lr_df = pd.DataFrame(list_results)
#
# lr_df.columns = ['epoch', 'train_loss', 'train_correct', 'validation_loss', 'validation_correct',
#                  'test_loss', 'test_correct']
# lr_df.to_csv('C:/Users/Usuario/Documents/practica_semeion/dl_lr-' + str(lr) + '-momentum-' + str(momentum) +
#              '-batch-' + str(batchsize) + '-seed-' + str(seed) + '.csv')
