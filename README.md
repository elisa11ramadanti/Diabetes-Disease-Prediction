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
    
---

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

4. Penanganan Missing Values dan Duplikasi  
   - Dalam kasus ini, kita memastikan bahwa tidak ada missing values (nilai kosong) atau duplikasi yang dapat mempengaruhi analisis atau pelatihan model.

5. Mengatasi Outliers
   
   - Outlier adalah nilai ekstrem yang dapat mempengaruhi kinerja model machine learning. Pada tahap ini, kita mengidentifikasi dan menangani outlier menggunakan metode Interquartile Range (IQR).
   - IQR digunakan untuk mendeteksi outlier dengan menghitung nilai ambang batas bawah dan atas berdasarkan kuartil pertama (Q1) dan kuartil ketiga (Q3).

6. Penanganan Data Tidak Seimbang
     - Dalam dataset ini, kita mungkin memiliki distribusi kelas yang tidak seimbang, misalnya, jumlah pasien yang terdiagnosis diabetes jauh lebih sedikit dibandingkan yang tidak. Untuk menangani ini, kita menggunakan teknik SMOTE (Synthetic Minority Over-sampling Technique).
     - SMOTE bekerja dengan membuat data sintetis dari kelas minoritas untuk menyeimbangkan jumlah data antara kelas-kelas tersebut.

7. Pembagian Data (Train-Test Split)
   
    Setelah data diproses, kita membagi dataset menjadi dua bagian utama:
     - Training Set (80%): Digunakan untuk melatih model.
     - Testing Set (20%): Digunakan untuk menguji akurasi model yang sudah dilatih. Pembagian ini memastikan bahwa model diuji pada data yang tidak terlihat selama pelatihan.

8. Feature Scaling (Normalisasi)

   - melakukan normalisasi fitur numerik. Ini penting karena beberapa algoritma machine learning sensitif terhadap skala fitur. Misalnya, jika ada fitur dengan rentang nilai yang sangat besar dan kecil, itu bisa membuat model kesulitan untuk menemukan pola yang baik.
   - Untuk ini, menggunakan StandardScaler, yang mengubah fitur agar memiliki rata-rata 0 dan deviasi standar 1, membuat fitur berada pada skala yang seragam.

---

## 🤖 Modeling
Pada tahap ini, dilakukan pemodelan menggunakan empat algoritma boosting yang populer: **XGBoost**, **LightGBM**, **CatBoost**, dan **Gradient Boosting**. Keempat model ini diuji dan dievaluasi menggunakan metrik akurasi, precision, recall, dan F1-score untuk menilai performa dalam memprediksi risiko diabetes.

### XGBoost

XGBoost (Extreme Gradient Boosting) adalah algoritma boosting berbasis pohon keputusan yang membangun model secara bertahap. Setiap model baru mencoba memperbaiki kesalahan dari model sebelumnya. XGBoost cenderung memiliki performa yang sangat baik, terutama pada dataset yang kompleks.

| Metrik       | Kelas 0 (Negatif) | Kelas 1 (Positif) |
|--------------|-------------------|-------------------|
| **Akurasi**  | 67.6%             | -                 |
| **Precision**| 0.69              | 0.66              |
| **Recall**   | 0.63              | 0.72              |
| **F1-Score** | 0.66              | 0.69              |


**Confusion Matrix:**

![Confusion Matrix XGBoost](https://github.com/user-attachments/assets/8d1ace8c-3f88-4962-adcd-822d3454b6a2)

---

### LightGBM

LightGBM adalah implementasi dari gradient boosting yang dikembangkan oleh Microsoft dan lebih cepat dalam memproses data besar.

| Metrik       | Kelas 0 (Negatif) | Kelas 1 (Positif) |
|--------------|-------------------|-------------------|
| **Akurasi**  | 68.3%             | -                 |
| **Precision**| 0.70              | 0.67              |
| **Recall**   | 0.65              | 0.72              |
| **F1-Score** | 0.67              | 0.69              |

**Confusion Matrix:**

![Confusion Matrix LightGBM](https://github.com/user-attachments/assets/e71d7dda-32f2-4e31-ac90-afbf4896680d)

---

### CatBoost

CatBoost adalah algoritma boosting lain yang dirancang untuk menangani data kategorikal tanpa perlu encoding khusus.

| Metrik       | Kelas 0 (Negatif) | Kelas 1 (Positif) |
|--------------|-------------------|-------------------|
| **Akurasi**  | 67.2%             | -                 |
| **Precision**| 0.71              | 0.65              |
| **Recall**   | 0.59              | 0.76              |
| **F1-Score** | 0.64              | 0.70              |

**Confusion Matrix:**

![Confusion Matrix CatBoost](https://github.com/user-attachments/assets/b385149b-f0b1-4efd-9eab-190f74c8f8e9)

---

### Gradient Boosting

Gradient Boosting adalah metode boosting klasik yang biasanya lebih lambat dibandingkan dengan XGBoost dan LightGBM.

| Metrik       | Kelas 0 (Negatif) | Kelas 1 (Positif) |
|--------------|-------------------|-------------------|
| **Akurasi**  | 61.2%             | -                 |
| **Precision**| 0.64              | 0.59              |
| **Recall**   | 0.50              | 0.72              |
| **F1-Score** | 0.56              | 0.65              |

**Confusion Matrix:**


![Confusion Matrix Gradient Boosting](https://github.com/user-attachments/assets/37b35776-6b61-4f69-993c-24d45f25d937)

---

### Hyperparameter Tuning

Hyperparameter tuning dilakukan menggunakan **GridSearchCV** untuk masing-masing model untuk menemukan parameter terbaik yang dapat meningkatkan performa model.

**Hasil Perbandingan Akurasi Setelah GridSearchCV**

| Model         | Akurasi   |
|---------------|-----------|
| **XGBoost**   | 65.5%     |
| **LightGBM**  | 69.4%     |
| **CatBoost**  | 71.6%     |

---

## Evaluation

**Metrik Evaluasi:**  
- **Akurasi:** Digunakan untuk mengukur seberapa banyak prediksi yang benar.
- **Precision, Recall, dan F1-Score:** Digunakan untuk mengevaluasi keseimbangan antara prediksi benar (true positives), prediksi salah (false positives), dan prediksi yang terlewat (false negatives), terutama karena dataset tidak seimbang.
---

## Conclusion

Berdasarkan hasil evaluasi, **CatBoost** dengan parameter terbaik setelah GridSearchCV memberikan performa terbaik dengan akurasi mencapai **71.6%**. Secara keseluruhan, model-model boosting yang digunakan dalam proyek ini menunjukkan potensi yang baik untuk memprediksi risiko diabetes, meskipun masih ada ruang untuk perbaikan, terutama dalam meningkatkan akurasi lebih lanjut dengan tuning yang lebih dalam atau penambahan fitur lain.

**Future Work:**
- Menambahkan fitur tambahan yang relevan untuk meningkatkan akurasi.
- Menggunakan algoritma lain untuk memverifikasi apakah ada model yang lebih baik.

---

## 📌 Referensi
- [WHO - Diabetes](https://www.who.int/news-room/fact-sheets/detail/diabetes)
- [Kaggle - Diabetes Prediction](https://www.kaggle.com/)

---

## 📫 Kontak
**Elisa Ramadanti**  
LinkedIn: [linkedin.com/in/elisa-ramadanti](https://www.linkedin.com/in/elisa-ramadanti)  
