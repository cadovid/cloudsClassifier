import numpy as np
import matplotlib.pyplot as pl

clouds_x = np.load("nubes_train.npy")
clouds_y = np.load("labels_train.npy")

# Busqueda de los elementos que nos interesa segun su clase
elementPosition = np.where(clouds_y == 4)
print(elementPosition)
for i in elementPosition:
    aa = clouds_y[i]

np.save('ValuesClass', aa)

# Guardamos en un array todos los datos del array original que nos interesa
for i in elementPosition:
    qq = clouds_x[i]
    print(qq.shape)

np.save('searchedClass', qq)

# # Mostramos las imagenes de los elementos
# for i in qq:
#     img = np.moveaxis(i,0,2)
#     pl.figure()
#     pl.imshow(img)
#     pl.show()

searched = np.load("searchedClass.npy")
values = np.load("ValuesClass.npy")
print(searched)
print(values)

# Mostramos las imagenes de los elementos
for i in searched:
    img = np.moveaxis(i,0,2)
    pl.figure()
    pl.imshow(img)
    pl.show()
