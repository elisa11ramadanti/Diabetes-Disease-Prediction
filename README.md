# 📊 Laporan Proyek Prediksi Diabetes dengan Machine Learning

## Diabetes-Disease-Prediction
**Disusun oleh: Elisa Ramadanti**

Proyek ini bertujuan untuk membangun model machine learning yang dapat memprediksi risiko penyakit diabetes berdasarkan data kesehatan individu. Dengan menggunakan beberapa algoritma klasifikasi dan teknik optimasi model, diharapkan hasil prediksi dapat membantu dalam deteksi dini dan pengambilan keputusan medis yang lebih baik.

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
Dataset yang digunakan berisi **1000 entri** dan **9 fitur** sebagai berikut:
- **Pregnancies**: Jumlah kehamilan
- **Glucose**: Kadar glukosa dalam darah
- **BloodPressure**: Tekanan darah
- **SkinThickness**: Ketebalan kulit
- **Insulin**: Kadar insulin dalam darah
- **BMI**: Indeks massa tubuh
- **DiabetesPedigreeFunction**: Riwayat diabetes dalam keluarga
- **Age**: Usia
- **Diagnosis**: Klasifikasi diabetes (0 = Tidak, 1 = Ya)

Dataset ini diambil dari [Kaggle - Diabetes Prediction](https://www.kaggle.com/).
![image](https://github.com/user-attachments/assets/bcc49156-199d-42f8-bfcb-936faaa2f79d)

### 🔍 Exploratory Data Analysis (EDA)
- **Visualisasi Distribusi Data**: Menggunakan histogram dan boxplot untuk melihat distribusi masing-masing fitur.

![image](https://github.com/user-attachments/assets/f32aedde-d85b-4ff8-8d1e-5a97fb4e666a)

- **Pairplot**: Digunakan untuk menganalisis hubungan antar fitur.
  ![image](https://github.com/user-attachments/assets/102ec4fe-a8fe-4d21-aa2e-1f7fb86e994f)

- **Boxplot**: Digunakan untuk mendeteksi outlier menggunakan metode IQR.
  ![image](https://github.com/user-attachments/assets/f52704ac-8c8f-48e5-9212-196181d375cf)
  ![image](https://github.com/user-attachments/assets/3adb68ae-4fa4-418a-8f59-3e95ed27f4ac)



### ⚖️ Class Distribution
- Dataset awal memiliki distribusi kelas yang tidak seimbang, di mana kelas **"Tidak Diabetes"** lebih dominan dibandingkan dengan kelas **"Diabetes"**.
- Untuk mengatasi class imbalance, teknik **SMOTE** (Synthetic Minority Over-sampling Technique) digunakan.
<p align="center">
  <img src="https://github.com/user-attachments/assets/ae7dc936-36aa-401e-8a0c-2372a9611fdc" alt="Class Distribution Before" width="40%" style="margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/ddd9010f-b475-4c78-88f8-977365125ebb" alt="Class Distribution After" width="40%">
</p>
  

---

## ⚙️ Data Preparation
1. **Handling Missing Values**:  
   - Dataset tidak memiliki nilai kosong, sehingga tidak diperlukan imputasi.
  
2. **Outlier Removal**:  
   - Outlier diatasi menggunakan metode **IQR (Interquartile Range)** untuk fitur numerik seperti Insulin dan BMI.

3. **Feature Scaling**:  
   - **StandardScaler** digunakan untuk menormalisasi fitur numerik agar memiliki mean 0 dan standar deviasi 1.

4. **Handling Imbalanced Data**:  
   - **SMOTE** digunakan untuk menyeimbangkan distribusi kelas target.

5. **Data Splitting**:  
   - Dataset dibagi menjadi **data latih (80%)** dan **data uji (20%)**.

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
