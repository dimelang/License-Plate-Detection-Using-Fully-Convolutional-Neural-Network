import Data_helper
import Evaluasi
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

loss = []
val_loss = []

# Muat kembali nilai loss dan val_loss yg sudah tersimpan sebelumnya
with open('Result/History_Train.pkl', 'rb') as f:
    log = pickle.load(f)
    for i in range(len(log)):
        # Dapatkan nilai loss terendah pada fold ke-i
        loss.append(np.amin(log[i].get('loss')))
        # Dapatkan nilai val_loss terendah pada fold ke-i
        val_loss.append(np.amin(log[i].get('val_loss')))

# Mencari nilai val_loss terkecil dari seluruh fold pelatihan
indeks = np.where(val_loss == np.amin(val_loss))[0]

# Load bobot sesuai dengan nilai yang terkandung dalam variabel indeks
# Bobot model yang menghasilkan nilai val_loss terkecil dari seluruh pelatihan
predModel = load_model('Result/Fold'+str(indeks[0]+1)+'.hdf5')

# Memuat data uji
X_test, y_test, yTest_Label = Data_helper.loadDataset("test")

# Prediksi citra heatmap
pred = predModel.predict(X_test)

# Evaluasi hasil prediksi
loss, AP = Evaluasi.evaluate(pred, y_test, yTest_Label, 0.50)
