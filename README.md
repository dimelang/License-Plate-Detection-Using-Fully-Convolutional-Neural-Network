## License Plate Detection Using Fully Convolutional Neural Network
---
Pemodelan _deep learning_ yang bertujuan untuk  deteksi plat nomor kendaraan pada sebuah citra. Arsitektur yang digunakan diadopsi dari jaringan [FCRN](https://www.tandfonline.com/doi/abs/10.1080/21681163.2016.1149104) dan dimodifikasi untuk meningkatkan kemampuan/performa jaringan.

##### Data
Data yang dipakai, diambil dari rekaman CCTV

##### Prapemrosesan Data
- Ekstrak frame
- Crop citra
- Konversi ke dalam citra
- Anotasi menggunakan [labelme](https://github.com/wkentaro/labelme)
- Resize citra
- Generate citra _heatmap_

##### Modifikasi :
- Modifikasi layer
- Menambahkan [Batch norm](https://arxiv.org/abs/1502.03167)
- Menambahkan [Skip connection](https://arxiv.org/abs/1505.04597)
- Mengganti metode upsampling

##### Training
Jaringan dilatih menggunakan GPU NVIDIA 1080Ti dan memori sebesar 16Gb. Pelatihan jaringan berdasarkan pada K-Fold Cross Validation. Setiap bobot jaringan yang mampu menghasilkan nilai val_loss terkecil akan disimpan sehingga pada akhir pelatihan akan tercipta sejumlah K bobot jaringan terbaik. 

##### Pengujian
Sejumlah bobot jaringan terbaik ditinjau kembali dengan mengamati nilai val_loss. Bobot jaringan dengan nilai val_loss terkecil dimanfaatkan untuk mengujia data uji.

##### Performa
Perbandingan performa terhadap ukuran citra yang berbeda. Metrik evaluasi menggunakan Average Precision (AP) yang meniru [Paddila](https://github.com/rafaelpadilla/Object-Detection-Metrics).
| Ukuran citra | AP | Waktu (detik) |
| ------ | ------ | ------ |
|96⁡×⁡96|77.86%|0.045|
|192⁡×192|93.79%|0.055|
|320⁡×320|96.45%|0.080|

Perbandingan terhadap [Faster RCNN](https://arxiv.org/abs/1506.01497)
| Ukuran citra | AP | Waktu (detik) |
| ------ | ------ | ------ |
|96⁡×⁡96|96.45%|0.80|
|960⁡×960|77.72%|3.198|

##### Pustaka
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [OpenCV](https://opencv.org/)
- [H5py](https://www.h5py.org/)
- [Matplotlib](https://matplotlib.org/)