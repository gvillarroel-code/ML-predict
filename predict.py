import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras


datos_csv = pd.read_excel('./predict.xlsx',header=0)
datos_csv.columns = ['A', 'B']

mylist_x=[]
mylist_y=[]

for d in range(len(datos_csv)):
     mylist_x.append([datos_csv['A'][d]])
     mylist_y.append([datos_csv['B'][d]])

xs = np.array(mylist_x, dtype=float)
ys = np.array(mylist_y, dtype=float)

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

#xs = np.array([1.0, 2.0, 7.0, 3.0, 11.0, 6.0,4], dtype=float)
#ys = np.array([2.29, 3.57, 10, 4.86, 15.14, 8.71,6.14], dtype=float)


model.fit(xs, ys, epochs=100)

print("-----------------------")
print("\n PREDICCION: " + str(model.predict([17.0])) + "\n")

