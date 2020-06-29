import torch
import torchvision

import time
import pandas as pd
import sklearn.utils
import torch
# torch.nn son clases que contienen parámetros. Finalmente llama a funciones en F.
# torch.nn.functional (F) son stateless (sin parámetros).
# Parámetros (pesos), si hacen falta, se tienen que almacenar de manera explícita.
# nn.Softmax es una función que se puede utilizar para calcular el softmax a partir de los logits
# LogSoftMax + NLLLloss = CrossEntropyLoss
# Se puede usar reduction="none" para ver el desglose por clases, pero en ese caso, no calcula la media para el minibatch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import matplotlib.pyplot as pl
import matplotlib.image as mpimg
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


# Para borrar la cache de la GPU, por si acaso
torch.cuda.empty_cache()

semeion_x = np.load("nubes_train.npy")
semeion_y = np.load("labels_train.npy")
semeion_validation_x = np.load("nubes_validation.npy")
semeion_validation_y = np.load("labels_validation.npy")
semeion_test_x = np.load("nubes_test.npy")
semeion_test_y = np.load("labels_test.npy")

# # Tenemos que reformatear los arrays para añadir un canal
# # (con imágenes en color tendrían que ser 3: RGB)
# semeion_x = semeion_x.values.reshape((-1, 1, 16, 16))
# semeion_validation_x = semeion_validation_x.values.reshape((-1, 1, 16, 16))
# semeion_test_x = semeion_test_x.values.reshape((-1, 1, 16, 16))

# ¿Podemos usar GPU? Si no, usar CPU, pero es más lento
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Usando device = ', device)

# Creamos ahora los tensores a partir de arrays numpy
semeion_x = torch.tensor(semeion_x, dtype=torch.float)
semeion_y = torch.tensor(semeion_y, dtype=torch.long)

semeion_validation_x = torch.tensor(semeion_validation_x, dtype=torch.float)
semeion_validation_y = torch.tensor(semeion_validation_y, dtype=torch.long)

semeion_test_x = torch.tensor(semeion_test_x, dtype=torch.float)
semeion_test_y = torch.tensor(semeion_test_y, dtype=torch.long)

# Para redes convolucionales no hace falta aplanar la entrada

#semeion_x = semeion_x.view(-1, 16*16)
#semeion_validation_x = semeion_validation_x.view(-1, 16*16)
#semeion_test_x = semeion_test_x.view(-1, 16*16)

# Aquí definimos la red de neuronas con 2 capas convolucionales (con sus correspondientes maxpool)
# y 2 capas fully connected al final


class NetConv(nn.Module):
    # Aquí definimos las capas de la red
    def __init__(self):
        super(NetConv, self).__init__()
        # Capa convolucional con 3 canal de entrada (la imagen en RGB) y 15 mapas de características a la salida, con filtros/kernels de 3x3
        self.conv1 = nn.Conv2d(3, 15, kernel_size=4)
        # Capa de max-pooling con filtros de 2x2 (y stride=2 por omisión)
        self.mp = nn.MaxPool2d(2)
        # Capa fully-connected (dense) con 8 entradas y 10 neuronas ocultas
        # Nota: el 8 hay que calcularlo a mano, y son las salidas de la capa anterior
        # Yo uso el truco de poner la variable visualiza = True y usar el código de la celda 17
        # También podemos calcularlo:
        #  la entrada es de 16x16
        #  dado que el filtro es de 3x3, con stride = 1 y sin padding, cada mapa será (16-3+1)x(16-3+1)=14x14
        #  el maxpool reduce dimensionalidad a la mitad. Cada mapa será 7x7. Hay tres mapas
        #  por tanto el aplanamiento dará 7x7x3 = 147
        # El número de salidas es 10 porque hay 10 clases
        self.fc = nn.Linear(55815, 12)
        # Métodos para inicializar los pesos, aunque no siempre mejorar a los métodos por omisión
        # Lo de nonlinearity='relu' es porque pytorch asume una 'leaky_relu' por omisión
        # para el bias, se suele dejar la inicialización por omisión
        torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.fc.weight)

    # Aquí definimos la función matemática que representa la red (conexiones entre capas)
    # Nótese que la red no procesa datos individuales, sino minibatches completos
    def forward(self, x):
        # in_size contiene el número de elementos en el minibatch
        in_size = x.shape[0]
        if (visualiza): print('A la entrada: ', x.shape, 'Tamaño del minibatch:', in_size)
        x = F.relu(self.mp(self.conv1(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        if (visualiza): print('Flattened tensor: ', x.shape)
        x = self.fc(x)
        if (visualiza): print(x.shape)
        # output = torch.log_softmax(x, dim=1)
        output = x
        if (visualiza): print(output.shape)
        return (output)


# PARÁMETROS
# Batchsize, típicamente 2^n
batchsize = 64
# Learning rate: 0.1, 0.01, 0.001 ...
lr = 0.01
momentum = 0.6
# La inicialización de los pesos es aleatoria, y esta es la semilla
seed = 1

# Aquí creamos el modelo (RED) inicial
np.random.seed(seed)
torch.manual_seed(seed)
model = NetConv().to(device)
print(model)

# Usamos utilidades de Pytorch para recorrer los conjuntos de entrenamiento, validación y test
# TensorDataset define el conjunto de datos
train_ds = TensorDataset(semeion_x, semeion_y)
validation_ds = TensorDataset(semeion_validation_x, semeion_validation_y)
test_ds = TensorDataset(semeion_test_x, semeion_test_y)

# Y el dataloader define la manera de recorrerlo
# En entrenamiento, los datos se van a recorrer en minibatches
# Shuffle=True para que en cada epoch, se reordenen todos los datos
train_dl = DataLoader(train_ds, batch_size=batchsize, shuffle=True)
# En validación y test meteremos todos los datos en el mismo minibatch (1)
validation_dl = DataLoader(validation_ds, batch_size=len(validation_ds))
test_dl = DataLoader(test_ds, batch_size=len(test_ds))

# Esto sirve para visualizar la red con el primer minibatch
# cuyas entradas van a estar en xb y las salidas en yb

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

# Elegimos como optimizador el más básico: Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# Otra posibilidad sería Adam
# optimizer = torch.optim.Adam(model.parameters(), lr = lr)

list_results = []

start = time.time()
for epoch in range(0, 200):
    # PARTE DE ENTRENAMIENTO
    # Con esto ponemos al modelo en modo entrenamiento
    # Es decir, se calculan gradientes y se actualizan pesos
    model = model.train()

    train_loss = train_correct = 0
    for xb, yb in train_dl:
        # Importante: hay que poner a cero los gradientes, si no, se suman
        optimizer.zero_grad()
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss = criterion(output, yb)
        # item es para extraer del grafo un 0-tensor en un scalar numpy
        # si fuera un 1-tensor o mas, habría que usar .cpu().detach().numpy()
        train_loss += loss.item()
        predicted = torch.max(output.data, dim=1)[1]
        train_correct += (predicted == yb).sum().item()
        loss.backward()
        optimizer.step()
    # Hacemos la suposición de que todos los batches tienen el mismo número de
    # elementos (que para el último batch puede no ser cierto).
    # En validación y en test el conjunto está en el mismo batch, así que en ese caso
    # la suposición es correcta.
    train_loss /= float(len(train_dl))
    train_correct /= float(len(train_ds))

    # PARTE DE VALIDACIÓN
    # Con esto ponemos al modelo en modo evaluación
    # Es decir, la red se usa para hacer predicciones pero no se calculan gradientes.
    model = model.eval()
    validation_loss = validation_correct = 0
    # with torch.no_grad() se evita también el cálculo de gradientes
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

    # PARTE DE TEST
    # Con esto ponemos al modelo en modo evaluación
    # Es decir, la red se usa para hacer predicciones pero no se calculan gradientes.
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

    # Imprime resultados cada 100 epochs
    if (epoch % 10 == 0):
        print('Epoch: {} TrLoss: {} Tr: {} ValLoss: {} Val: {} TsLoss: {} Ts: {}'.format(*result[0:3], *result[3:5],
                                                                                         *result[5:]))

end = time.time()
print('tiempo', end - start)

# Seleccionamos el número de epochs allí donde se alcance el mínimo valor en validación
list_res_np = np.array(list_results)
mejor_epoch_validacion = np.argmin(np.array(list_res_np)[:,3])
print('El mejor valor de la Loss en validación se obtuvo en el epoch: ',mejor_epoch_validacion)
print('Y las tasas de acierto en entrenamiento, validación y test eran:', list_res_np[mejor_epoch_validacion,[2,4,6]])

from matplotlib import pyplot as plt

# Aquí ploteamos las tasas de acierto en entrenamiento, validación y test
plt.figure()
plt.subplots_adjust(hspace=0.5)

plt.subplot(2,1,1)
plt.plot(list_res_np[:,0], list_res_np[:,2], label="Train")
plt.plot(list_res_np[:,0], list_res_np[:,4], label="Validation")
plt.plot(list_res_np[:,0], list_res_np[:,6], label="Test")
plt.legend(loc="lower right")
plt.title('Tasa de aciertos')

# Y aquí, las loss
plt.subplot(2,1,2)
plt.plot(list_res_np[:,0], list_res_np[:,1], label="Train")
plt.plot(list_res_np[:,0], list_res_np[:,3], label="Validation")
plt.plot(list_res_np[:,0], list_res_np[:,5], label="Test")
plt.legend(loc="upper right")
plt.title('Loss')
plt.show()

# # Por si queremos salvar los resultados de esta ejecución en un fichero
# # Si utilizais esto, cambiad el código según vuestras necesidades
# lr_df = pd.DataFrame(list_results)
#
# lr_df.columns = ['epoch', 'train_loss', 'train_correct', 'validation_loss', 'validation_correct',
#                  'test_loss', 'test_correct']
# lr_df.to_csv('C:/Users/Usuario/Documents/practica_semeion/dl_lr-' + str(lr) + '-momentum-' + str(momentum) +
#              '-batch-' + str(batchsize) + '-seed-' + str(seed) + '.csv')