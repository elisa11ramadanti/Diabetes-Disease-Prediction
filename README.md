# 📊 Laporan Proyek Prediksi Diabetes dengan Machine Learning

## Diabetes-Disease-Prediction
**Disusun oleh: Elisa Ramadanti**

Proyek ini bertujuan untuk membangun model prediksi risiko diabetes berdasarkan dataset yang terdiri dari beberapa fitur medis. Model yang digunakan adalah empat algoritma boosting yang populer, yaitu XGBoost, LightGBM, CatBoost, dan Gradient Boosting. Setiap model diuji dan dievaluasi menggunakan metrik seperti akurasi, precision, recall, F1-score, serta confusion matrix.

---

## 📌 Domain Proyek
Diabetes merupakan salah satu penyakit kronis yang banyak diderita masyarakat di seluruh dunia. Menurut [WHO](https://www.who.int/news-room/fact-sheets/detail/diabetes), prevalensi diabetes terus meningkat secara global, menjadikannya masalah kesehatan yang mendesak. Deteksi dini terhadap risiko diabetes sangat penting untuk mencegah komplikasi yang lebih serius, seperti penyakit jantung, stroke, dan kerusakan organ lainnya.

**Prediksi diabetes secara akurat dapat membantu dalam:**
- Deteksi dini risiko diabetes.
- Pencegahan komplikasi yang lebih serius.
- Membantu tenaga medis dalam pengambilan keputusan yang lebih baik.

**Tujuan utama:**  
Mengembangkan model machine learning yang mampu memprediksi risiko diabetes secara akurat berdasarkan data kesehatan individu.

---

## 📊 Business Understanding
### 📌 Problem Statements
- Bagaimana mengidentifikasi individu dengan risiko diabetes menggunakan data kesehatan?
- Bagaimana meningkatkan akurasi prediksi diabetes menggunakan model machine learning yang tepat?

### 🎯 Goals
- Membuat model machine learning yang mampu memprediksi risiko diabetes dengan akurasi tinggi.
- Menentukan algoritma terbaik melalui evaluasi model menggunakan beberapa metrik evaluasi.

### 🛠️ Solution Statements
- Menggunakan beberapa algoritma klasifikasi, terutama algoritma ensemble seperti **XGBoost**, **LightGBM**, **Gradient Boosting**, dan **CatBoost**.
- Menerapkan **hyperparameter tuning** menggunakan GridSearchCV untuk mengoptimalkan performa model.

---

## 📁 Data Understanding
Dataset yang digunakan adalah Diabetes Prediction yang diperoleh dari Kaggle. Dataset ini bertujuan untuk memprediksi diabetes berdasarkan faktor risiko yang relevan.

1. **Jumlah Data:**  memiliki **1000 baris dengan 9 Kolom** sebagai berikut:  
  - **Pregnancies**: Jumlah kehamilan  
  - **Glucose**: Kadar glukosa dalam darah  
  - **BloodPressure**: Tekanan darah  
  - **SkinThickness**: Ketebalan lipatan kulit  
  - **Insulin**: Kadar insulin  
  - **BMI**: Indeks massa tubuh  
  - **DiabetesPedigreeFunction**: Fungsi silsilah diabetes (mengindikasikan riwayat keluarga)  
  - **Age**: Usia  
  - **Diagnosis**: Label klasifikasi dengan nilai:  
    - **0** = Tidak Diabetes  
    - **1** = Diabetes  

2. **Kondisi Data:** Dataset ini memiliki ketidakseimbangan kelas pada kolom Diagnosis, dengan jumlah label "1" (diabetes) dan "0" (tidak diabetes) yang tidak merata. 
  - Ini perlu diperhatikan untuk menghindari bias pada model prediksi yang akan dibangun.

3. **Sumber Data:** Dataset ini diambil dari [Kaggle - Diabetes Prediction](https://www.kaggle.com/).

4. **Eksplorasi Data:**  beberapa teknik visualisasi dan analisis eksploratori data akan dilakukan. Misalnya:

  - Distribusi Data: Visualisasi distribusi nilai untuk setiap fitur seperti BMI, Glucose, dan Age untuk memahami sebaran data.
  - Korelasi: Melakukan analisis korelasi antar fitur untuk melihat hubungan yang mungkin ada antar fitur, terutama dengan target label Diagnosis.
  - Imbalance Handling: Menganalisis ketidakseimbangan kelas dalam Diagnosis dan menerapkan teknik seperti SMOTE (Synthetic Minority Over-sampling Technique) untuk menangani masalah tersebut.
    

## ⚙️ Data Preparation
1. Memuat Data
    - Data dimuat dari file CSV ke dalam sebuah dataframe menggunakan pandas. Ini memungkinkan kita untuk melihat dan mengelola data dengan mudah.
  
2. Assessing Data 
   memeriksa informasi dasar dari dataset untuk memahami strukturnya, seperti:
    - Tipe data setiap kolom (df.info()).
    - Memeriksa nilai yang hilang atau null (df.isnull().sum()).
    - Memeriksa duplikasi data untuk memastikan tidak ada entri yang berulang.
    - Melihat statistik deskriptif dari data numerik untuk memeriksa distribusi data (df.describe()).

3. Exploratory Data Analysis (EDA)
   EDA dilakukan untuk memahami pola dalam data, seperti:
   - Distribusi Fitur Numerik: Menggunakan visualisasi histogram untuk melihat distribusi dari fitur numerik, yang dapat memberikan informasi tentang apakah ada fitur yang terdistribusi normal atau terpengaruh oleh outlier.
   - Korelasi Antar Fitur: Menganalisis korelasi antar fitur menggunakan matriks korelasi. Ini penting untuk mengetahui apakah ada fitur yang sangat berkorelasi, yang bisa mempengaruhi kinerja model.

5. Penanganan Missing Values dan Duplikasi  
   - Dalam kasus ini, kita memastikan bahwa tidak ada missing values (nilai kosong) atau duplikasi yang dapat mempengaruhi analisis atau pelatihan model.

6. Mengatasi Outliers  
    Outlier adalah nilai ekstrem yang dapat mempengaruhi kinerja model machine learning. Pada tahap ini, kita mengidentifikasi dan menangani outlier menggunakan metode Interquartile Range (IQR).
   - IQR digunakan untuk mendeteksi outlier dengan menghitung nilai ambang batas bawah dan atas berdasarkan kuartil pertama (Q1) dan kuartil ketiga (Q3).

7. Penanganan Data Tidak Seimbang
Dalam dataset ini, kita mungkin memiliki distribusi kelas yang tidak seimbang, misalnya, jumlah pasien yang terdiagnosis diabetes jauh lebih sedikit dibandingkan yang tidak. Untuk menangani ini, kita menggunakan teknik SMOTE (Synthetic Minority Over-sampling Technique).

SMOTE bekerja dengan membuat data sintetis dari kelas minoritas untuk menyeimbangkan jumlah data antara kelas-kelas tersebut.

8. Pembagian Data (Train-Test Split)
Setelah data diproses, kita membagi dataset menjadi dua bagian utama:

Training Set (80%): Digunakan untuk melatih model.
Testing Set (20%): Digunakan untuk menguji akurasi model yang sudah dilatih. Pembagian ini memastikan bahwa model diuji pada data yang tidak terlihat selama pelatihan.

9. Feature Scaling (Normalisasi)
Langkah selanjutnya adalah melakukan normalisasi fitur numerik. Ini penting karena beberapa algoritma machine learning sensitif terhadap skala fitur. Misalnya, jika ada fitur dengan rentang nilai yang sangat besar dan kecil, itu bisa membuat model kesulitan untuk menemukan pola yang baik.

Untuk ini, kita menggunakan StandardScaler, yang mengubah fitur agar memiliki rata-rata 0 dan deviasi standar 1, membuat fitur berada pada skala yang seragam.

---

## 🤖 Modeling
Proses modeling dilakukan menggunakan beberapa algoritma machine learning:
1. **Gradient Boosting**
2. **CatBoost**
3. **LightGBM**
4. **XGBoost**

### 🔧 Hyperparameter Tuning
Hyperparameter tuning dilakukan menggunakan **GridSearchCV** untuk mengoptimalkan akurasi dan performa model.

- **XGBoost**:
  - Parameter default digunakan untuk baseline model.
  - Hyperparameter tuning dilakukan pada parameter `n_estimators`, `learning_rate`, dan `max_depth`.
  
- **LightGBM**:
  - Hyperparameter tuning dilakukan pada parameter `n_estimators`, `learning_rate`, `max_depth`, `num_leaves`, `subsample`, dan `colsample_bytree`.
  
- **CatBoost**:
  - Hyperparameter tuning dilakukan pada parameter `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, dan `border_count`.
  
### 📊 Kelebihan dan Kekurangan Algoritma
- **XGBoost**: Performa tinggi pada dataset tabular, namun lebih kompleks dalam implementasi.
- **LightGBM**: Cepat dan efisien, cocok untuk dataset besar.
- **CatBoost**: Mampu menangani fitur kategorikal secara otomatis, namun waktu pelatihan lebih lambat.

---

## 📏 Evaluation
Model dievaluasi menggunakan beberapa metrik:
- **Accuracy**: Persentase prediksi yang benar.
- **Precision**: Akurasi dari prediksi positif.
- **Recall**: Kemampuan model dalam menemukan semua kasus positif.
- **F1-Score**: Harmonik rata-rata precision dan recall untuk keseimbangan kinerja model.

### 🎯 Hasil Evaluasi:
Berikut adalah hasil evaluasi model sebelum dan setelah **Hyperparameter Tuning** dalam bentuk tabel:

| **Model**                 | **Baseline Accuracy** | **Tuning Accuracy** |
|----------------------------|-----------------------|---------------------|
| **XGBoost**                | 67.63%                | 65.47%              |
| **LightGBM**               | 68.35%                | 69.42%              |
| **Gradient Boosting**      | 61.15%                | -                   |
| **CatBoost**               | 67.27%                | 71.58%              |

---

## ✅ Conclusion
Proyek ini berhasil membangun model prediktif untuk deteksi dini diabetes menggunakan algoritma machine learning seperti **XGBoost**, **LightGBM**, dan **CatBoost**. Model terbaik adalah **CatBoost** dengan akurasi **71.58%** setelah hyperparameter tuning.

---

## 📌 Referensi
- [WHO - Diabetes](https://www.who.int/news-room/fact-sheets/detail/diabetes)
- [Kaggle - Diabetes Prediction](https://www.kaggle.com/)

---

## 📫 Kontak
**Elisa Ramadanti**  
LinkedIn: [linkedin.com/in/elisa-ramadanti](https://www.linkedin.com/in/elisa-ramadanti)  
