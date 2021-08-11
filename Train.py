import Data_helper
import Network
import Evaluasi
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Concatenate, concatenate, Reshape, Add, BatchNormalization, Cropping2D, UpSampling2D
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.layers.core import Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2DTranspose
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.python.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

from sklearn.model_selection import KFold

# Hyperparameter CNN
upsample = True  # False = Upsampling, True = Transposed
skip_connection = False  # False/True
Batch_size = 2  # 1,2,4,8,16
batch_norm = True  # True/False
lr = 0.01  # 0.1,0.01,0.001
iter = 100  # 50,100,150,200
filter = [32, 32, 64, 128, 512]  # [16,32,64,128,256],[32,64,128,256,512]
kernel = [5, 5]  # [3,3],[5,5],[7,7]

# Cross validation
kf = KFold(n_splits=5)
Fold = 0

history = []  # Menampung nilai loss dan val_loss di setiap fold pelatihan
score = []  # Menampung nilai Average Precision di setiap fold pelatihan

# Split data latih ke dalam 5 partisi (5 fold cross validation)
for train_index, val_index in kf.split(Data_helper.loadDataset("train", False)[0]):

    # Inisialisasi ukuran input (tinggi citra,lebar citra, dan channel citra)
    inputShape = Data_helper.loadDataset("train", False)[0].shape[1:]

    # Membangun jaringan FCN
    # Semua hyperparameter harus diisi !
    model = Network.makeModel(
        inputShape, kernel, filter, lr, skip_connection, upsample, batch_norm)

    # Menentukan lokasi dan nama bobot model yg akan disimpan
    checkpoint_name = 'Result/Fold'+str(Fold+1)+'.hdf5'

    # Strategi pelatihan
    # Bobot model terbaik (ditinjau dari nilai val_loss paling kecil) yang akan tersimpan
    checkpoint = ModelCheckpoint(
        checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Mulai pelatihan jaringan
    callbacks_list = [checkpoint]

    # Data validasi digunakan untuk mendapatkan val_loss. Berguna sebagai penentu  suatu bobot pada pelatihan fold ke-i tersimpan
    # Fit data latih
    hist = model.fit(Data_helper.loadDataset("train", True, train_index)[0], Data_helper.loadDataset("train", True, train_index)[1], batch_size=Batch_size,
                     validation_data=(Data_helper.loadDataset("train", True, val_index)[0], Data_helper.loadDataset("train", True, val_index)[1]), epochs=iter, callbacks=callbacks_list)

    # Menyimpan nilai loss dan val_loss dari bobot model yang tersimpan
    history.append(hist.history)

    # Bobot jaringan terbaik pada fold ke-i dimanfaatkan untuk menguji data validasi
    validationModel = load_model('Result/Fold'+str(Fold+1)+'.hdf5')

    # Memuat data validasi
    XVal, yVal, yVal_Label = Data_helper.loadDataset("train", True, val_index)

    # Prediksi citra heatmap dari data validasi menggunakan bobot model yang tersimpan sebelumnya
    predVal = validationModel.predict(XVal)

    # Evaluasi citra heatmap yang dihasilkan
    # Metrik pengukuran menggunakan Average Precision (AP)
    loss, AP = Evaluasi.evaluate(predVal, yVal, yVal_Label, 0.50)

    # Simpan score (AP) yang didapatkan
    score.append([loss, AP])

    # Reset kembali bobot jaringan
    tf.keras.backend.clear_session()
    Fold += 1

# Hitung rata-rata score (AP) validasi
avgScore = np.mean(score, axis=0)
print("Rata-rata Loss:", avgScore[0], ", ",
      "Rata-rata AP", "{:.4%}".format(avgScore[1]))

# Simpan seluruh nilai loss dan val_loss utk keperluan evaluasi dan pengujian data test
with open('Result/History_Train.pkl', 'wb') as f:
    pickle.dump(history, f)
