import numpy as np
import matplotlib.pyplot as pl

semeion_x = np.load("nubes_train.npy")
semeion_y = np.load("labels_train.npy")
# semeion_validation_x = np.load("nubes_validation.npy")
# semeion_validation_y = np.load("labels_validation.npy")
# semeion_test_x = np.load("nubes_test.npy")
# semeion_test_y = np.load("labels_test.npy")

# print(semeion_x.shape)
# print(semeion_y.shape)
# print(semeion_x)
# print(semeion_y)
# print(semeion_validation_x.shape)
# print(semeion_validation_y.shape)
# print(semeion_validation_y)
# print(semeion_test_x.shape)
# print(semeion_test_y.shape)
# print(semeion_test_y)

# Busqueda de los elementos que nos interesa segun su clase
elementPosition = np.where(semeion_y == 4)
print(elementPosition)
for i in elementPosition:
    aa = semeion_y[i]

np.save('ValuesClass', aa)

# Guardamos en un array todos los datos del array original que nos interesa
for i in elementPosition:
    qq = semeion_x[i]
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