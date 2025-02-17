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
- Menggunakan beberapa algoritma klasifikasi, terutama algoritma ensemble seperti **XGBoost**, **LightGBM**, dan **CatBoost**.
- Menerapkan **hyperparameter tuning** menggunakan GridSearchCV untuk mengoptimalkan performa model.

---

## 📁 Data Understanding
Dataset yang digunakan adalah Diabetes Prediction yang diperoleh dari Kaggle. Dataset ini bertujuan untuk memprediksi diabetes berdasarkan faktor risiko yang relevan.

**Informasi Data:**
1. **Jumlah Data:** Dataset terdiri dari **1000 baris** dan **9 kolom**, termasuk 8 fitur numerik dan 1 label target (`Diagnosis`).
2. **Kondisi Data:**
   - Dataset memiliki ketidakseimbangan kelas pada kolom Diagnosis, dengan jumlah label "1" (diabetes) sebanyak 306 dan label "0" (tidak diabetes) sebanyak 694.
   - Tidak ada missing values atau duplikat data.
   - Outlier ditemukan pada beberapa fitur numerik seperti `Glucose`, `BloodPressure`, dan `BMI`, yang kemudian ditangani menggunakan metode IQR.
3. **Sumber Data:** Dataset diambil dari [Kaggle - Diabetes Prediction](https://www.kaggle.com/datasets/mrsimple07/diabetes-prediction/data).
4. **Uraian Fitur:**

| **Fitur**                  | **Deskripsi**                                                                 |
|----------------------------|-------------------------------------------------------------------------------|
| Pregnancies                | Jumlah kehamilan                                                             |
| Glucose                    | Kadar glukosa dalam darah                                                    |
| BloodPressure              | Tekanan darah                                                                |
| SkinThickness              | Ketebalan lipatan kulit                                                      |
| Insulin                    | Kadar insulin                                                                |
| BMI                        | Indeks massa tubuh                                                           |
| DiabetesPedigreeFunction   | Fungsi silsilah diabetes (mengindikasikan riwayat keluarga)                  |
| Age                        | Usia                                                                         |
| Diagnosis                  | Label klasifikasi (0 = Tidak Diabetes, 1 = Diabetes)                         |

5. **Tipe Data:**  
   - Semua fitur numerik memiliki tipe data `float64` atau `int64`.  
   - Tidak ada fitur kategorikal yang memerlukan encoding.
     
6. **Exploratory Data Analysis (EDA):**
   - Distribusi Data: Visualisasi distribusi setiap fitur.
   - Korelasi Antar Fitur: Menggunakan heatmap untuk untuk melihat hubungan yang mungkin ada antar fitur, terutama dengan target label diagnosis
   - Distribusi Target (Diagnosis): Melihat keseimbangan kelas target.

---

## ⚙️ Data Preparation
**Tahapan Data Preparation:**
1. **Penanganan Outlier:** Menggunakan metode IQR untuk semua fitur numerik.
2. **Penanganan Ketidakseimbangan Data:** Menggunakan SMOTE untuk membuat data sintetis pada kelas minoritas.
3. **Pembagian Data:** Membagi dataset menjadi 80% training set dan 20% testing set.
4. **Feature Scaling:** Normalisasi fitur numerik menggunakan StandardScaler untuk memastikan semua fitur memiliki mean 0 dan standar deviasi 1.

---

## 🤖 Modeling
Pada tahap ini, dilakukan pemodelan menggunakan empat algoritma boosting yang populer: **XGBoost**, **LightGBM**, dan **CatBoost**. Ketiga model ini diuji dan dievaluasi menggunakan metrik akurasi, precision, recall, dan F1-score untuk menilai performa dalam memprediksi risiko diabetes.

### Cara Kerja Algoritma
1. **XGBoost:** Algoritma boosting berbasis pohon keputusan yang membangun model secara bertahap. Parameter default digunakan sebelum hyperparameter tuning.
2. **LightGBM:** Implementasi gradient boosting yang dikembangkan oleh Microsoft, lebih cepat dalam memproses data besar. Parameter default digunakan sebelum hyperparameter tuning.
3. **CatBoost:** Algoritma boosting yang dirancang untuk menangani data kategorikal tanpa perlu encoding khusus. Parameter default digunakan sebelum hyperparameter tuning.

### Hyperparameter Tuning
Hyperparameter tuning dilakukan menggunakan GridSearchCV untuk mencari kombinasi parameter terbaik yang dapat meningkatkan performa model. Parameter terbaik yang diperoleh adalah sebagai berikut:
1. **XGBoost:** `learning_rate=0.1, max_depth=7, n_estimators=200`
2. **LightGBM:** `colsample_bytree=0.9, learning_rate=0.2, max_depth=-1, n_estimators=300, num_leaves=50, subsample=0.8`
3. **CatBoost:** `border_count=32, depth=8, iterations=500, l2_leaf_reg=1, learning_rate=0.1`

---

## Evaluation
### **Metrik Evaluasi:**
- **Akurasi:** Rasio prediksi yang benar terhadap total prediksi.
- **Precision:** Proporsi prediksi positif yang benar dibandingkan dengan semua prediksi positif.
- **Recall:** Proporsi contoh positif yang berhasil diprediksi dengan benar.
- **F1-Score:** Kombinasi harmonik antara precision dan recall.

### **Hasil Evaluasi**
**Sebelum Menggunakan Hyperparameter Tuning**

| Model         | Akurasi | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) |
|---------------|---------|---------------|---------------|------------|------------|--------------|--------------|
| **XGBoost**   | 67.6%   | 0.69          | 0.66          | 0.63       | 0.72       | 0.66         | 0.69         |
| **LightGBM**  | 68.3%   | 0.70          | 0.67          | 0.65       | 0.72       | 0.67         | 0.69         |
| **CatBoost**  | 67.0%   | 0.71          | 0.65          | 0.59       | 0.76       | 0.64         | 0.70         |

**Sesudah Menggunakan Hyperparameter Tuning**

| Model         | Akurasi | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) |
|---------------|---------|---------------|---------------|------------|------------|--------------|--------------|
| **XGBoost**   | 65.5%   | 0.67          | 0.64          | 0.62       | 0.69       | 0.64         | 0.67         |
| **LightGBM**  | 69.4%   | 0.72          | 0.67          | 0.63       | 0.76       | 0.67         | 0.71         |
| **CatBoost**  | 71.6%   | 0.74          | 0.69          | 0.66       | 0.77       | 0.70         | 0.73         |

---

## Conclusion

Berdasarkan hasil evaluasi, CatBoost dengan parameter terbaik setelah hyperparameter tuning memberikan performa terbaik dibandingkan model lainnya. Berikut adalah ringkasan temuan utama dari proyek ini:

1. Perbandingan Model
      - CatBoost : Memberikan akurasi tertinggi sebesar 71.6% , diikuti oleh LightGBM dengan akurasi 69.4% dan XGBoost dengan akurasi 65.5% .

2. Performa Metrik Evaluasi
      - Precision dan Recall :
            1. CatBoost menunjukkan precision yang lebih tinggi untuk kelas 0 (0.74 ) dan recall yang baik untuk kelas 1 (0.77 ), menjadikannya model yang lebih sensitif terhadap risiko diabetes.
            2. LightGBM dan XGBoost memiliki precision dan recall yang seimbang untuk kedua kelas, tetapi tidak seoptimal CatBoost.
      - F1-Score :
            - CatBoost mencapai F1-Score tertinggi untuk kedua kelas, yaitu 0.70 (kelas 0) dan 0.73 (kelas 1), yang menunjukkan keseimbangan antara precision dan recall.
3. Penanganan Ketidakseimbangan Data
      - Teknik SMOTE berhasil meningkatkan recall untuk kelas minoritas (kelas 1), yang membantu model mengenali lebih banyak kasus diabetes tanpa mengorbankan performa pada kelas mayoritas.
4. Rekomendasi
      - Model Terbaik : CatBoost adalah pilihan terbaik untuk prediksi risiko diabetes berdasarkan akurasi dan metrik evaluasi lainnya.
      - Hyperparameter Tuning : Langkah ini sangat penting untuk meningkatkan performa model. Parameter terbaik yang diperoleh melalui GridSearchCV secara signifikan meningkatkan akurasi model.
5. Future Work :
      - Penambahan Fitur : Menambahkan fitur tambahan seperti riwayat medis keluarga atau gaya hidup dapat meningkatkan akurasi model.
      - Teknik Optimasi Lanjutan : Menggunakan teknik optimasi lain seperti RandomizedSearchCV atau Bayesian Optimization untuk eksplorasi ruang parameter yang lebih luas.
      - Algoritma Alternatif : Menguji algoritma lain seperti Random Forest atau Neural Networks untuk memverifikasi apakah ada model yang lebih baik.
---

## 📌 Referensi
- [WHO - Diabetes](https://www.who.int/news-room/fact-sheets/detail/diabetes)
- [Kaggle - Diabetes Prediction](https://www.kaggle.com/datasets/mrsimple07/diabetes-prediction/data)

---

## 📫 Kontak
**Elisa Ramadanti**  
LinkedIn: [linkedin.com/in/elisa-ramadanti](https://www.linkedin.com/in/elisa-ramadanti)
