# Laporan Proyek Machine Learning - Irfan
Nama: Irfan Saputra Nasution
Asal: Pekanbaru, Riau

## Domain Proyek
Domain yang dipilih untuk projek Machine Learning adalah **Pertanian**, dengan judul Predictiver Analytics - Banana Quality

### Latar Belakang
Indonesia, sebagai negara tropis dengan keanekaragaman hayati yang tinggi, memiliki potensi besar dalam produksi pisang. Menurut data Kementrian Pertanian, Indonesia merupakan salah satu produsen pisang terbesar di dunia, dengan produksi mencapai 7,26 juta ton pada tahun 2020 [1].

Salah satu tantangan utama dalam industri pisang adalah dengan mempertahankan kualitas produk. Faktor-faktor seperti ukuran buah yang tidak optimal, tingkat kematangan yang tidak merata, dan kehilangan kerenyahan dapat menurunkan kualitas pisang. Penurunan ini berdampak pada kerugian ekonomi yang signifikan bagi petani dan distributor [2].

Penerapan predictive analytics dalam industri pisang Indonesia tidak hanya berpotensi meningkatkan kualitas produk, tetapi juga dapat mendukung keberlanjutan. Dengan meminimalkan  kerugian dan mengoptimalkan penggunaan sumber daya, teknologi ini dapat membantu mengurangi dampak lingkungan dari produksi pisang [3].

## Business Understanding
Pengembangan model prediksi kualitas pisang ini memiliki potensi besar dalam memberikan manfaat bagi petani dan distributor. Dengan menggunakan model ini, kualitas panen dapat ditingkatkan, nilai jual pisang dapat diperbaiki, dan kepercayaan konsumen dapat diperkuat. Prediksi yang akurat juga membantu petani dalam proses pemilahan apel serta menentukan harga jual yang lebih tepat.

### Problem Statement
Berdasarkan latar belakang di atas, berikut merupakan rincian masalah yang dapat diselesaikan pada projek ini:
- Bagaimana cara mengembangkan model machine learning yang mampu memprediksi kualitas pisang menggunakan data visual dan sensorik?
- Jenis model apa yang cenderung memberikan akurasi terbaik dalam prediksi ini?
- Bagaimana model tersebut dapat mendukung petani dan distributor dalam meningkatkan kualitas dan harga jual pisang?

### Goal
Tujuan dari projek ini adalah:
- Membuat sebuah model machine learning yang mampu memprediksi kualitas pisang menggunakan data visual dan sensorik
- Dengan membuat perbandingan beberapa algoritma model untuk menemukan akurasi terbaik dalam memprediksi kualitas pisang
- Dengan mengembangkan aplikasi yang dapat digunakan dengan mudah untuk membantu petani serta distributor dalam menggunakan model machine learning dalam memprediksi kualitas pisang.

### Solution Statements
- Melakukan proses data cleaning dengan cara proses checking data yang melihat satu-satu kolom tabel data yang memeriksa duplikat, data kosong, dan lain-lain.
- Menganalisis persebaran distribusi dari masing-masing variabel baik itu kategorikal atau numerikal.
- Memahami data demhan dilakukan dengan visualisasi dan mengetahui korelasi matriks antar fitur serta mendeteksi outlier
- Membuat beberapa variasi model untuk mendapatkan model yang paling baik dari beberapa model yang telah dibuat untuk prediksi kualitas pisang. Diantaranya adalah:
  * Random Forest adalah ensemble learning method yang mengoperasikan dengan membangun sejumlah besar pohon keputusan selama fase pelatihan. Setiap pohon di "forest" dibangun dari sampel boostrap dari data pelatihan. [4]
  * Logistic Regression, meskipun namanya mengandung kata "regression", sebernarnya adalah metode klasifikasi. Ini menggunakan fungsi logistik (juga dikenal sebagai fungsi sigmoid) untuk memetakan kombinasi linear dari fitur input ke probabilitas output [5].
  * SVC bekerja dengan menemukan hyperplane dalam ruang N-dimensi (N adalah jumlah fitur) yang secara jelas memisahkan titik-titik data ke dalam kelas-kelas yang berbeda. SVC berusaha untuk memaksimalkan margin antara kelas-kelas [6].
  * Multi-layer Perceptron (MLP) adalah jaringan saraf tiruan yang terdiri dari setidaknya tiga lapisan node: lapisan input, satu atau lebih lapisan tersembunyi, dan lapisan output. Setiap node, kecuali node input, menggunakan fungsi aktivasi nonlinear. [7]
  * CatBoost adalah implementasi gradient boosting yang menangani fitur kategorikal secara efisien. Ini menggunakan teknik ordered boosting dan symmetric trees. Dalam ordered boosting, CatBoost membangun beberapa model, masing-masing dilatih pada subset data yang berbeda. Ini membantu mengurangi overfitting dengan menghindari kebocoran target [8].
  * AdaBoost (Adaptive Boosting) adalah algoritma boosting yang membangun model kuat dengan menggabungkan banyak weak learners. Pada setiap iterasi, AdaBoost meningkatkan bobot sampel yang salah diklasifikasikan dan menurunkan bobot yang benar diklasifikasikan [9].
  * Extra Trees (Extremely Randomized Trees) adalah varian dari Random Forest yang memperkenalkan randomisasi tambahan dalam cara pohon keputusan dibangun. Saat membangun pohon keputusan, Extra Trees memilih titik split secara acak untuk setiap fitur yang dipertimbangkan, alih-alih mencari split terbaik. Ini berbeda dengan Random Forest yang mencari split optimal di antara subset acak fitur [10].
  * Gradient Boosting membangun model aditif secara bertahap. Pada setiap tahap, ia menambahkan weak learner baru yang meminimalkan fungsi loss [11].
  * Hist Gradient Boosting adalah implementasi Gradient Boosting yang menggunakan histogram untuk percepatan. Ini mirip dengan LightGBM [12].
  * XGBoost (eXtreme Gradient Boosting) adalah implementasi gradient boosting yang sangat dioptimalkan. Ia menggunakan model aditif yang serupa dengan Gradient Boosting tradisional, tetapi dengan beberapa penyempurnaan [13].
  * XGBRF (XGBoost Random Forest) adalah hybrid antara XGBoost dan Random Forest. Ini menggabungkan teknik bagging dari Random Forest dengan kekuatan XGBoost sebagai base learner [14].

## Data Understanding
### Exploratory Data Analysis (EDA) - Deskripsi Variabel
#### Informasi Dataset

| Jenis | Keterangan |
| ------ | ------ |
| Title | _Banana Quality_ |
| Source | [Kaggle](https://www.kaggle.com/datasets/l3llff/banana) |
| Maintainer | [l3LlFF](https://www.kaggle.com/l3llff) |
| License | Other (specified in description) |
| Visibility | Publik |
| Tags | _Earth and Nature, Education, Food, Data Visualization, Exploratory Data Analysis, Binary Classification_ |
| Usability | 10.00 |

Berikut adalah informasi pada dataset. Data yang digunakan dalam pembuatan model merupakan data primer, yang disediakan secara publik di kaggle dengan nama dataset yaitu: _Banana Quality_

![image](https://github.com/user-attachments/assets/6f6119e0-c633-4282-9453-71d5e1621685)
Tabel 1. Exploratory Data Analysis (EDA) Deskripsi Variabel

Dilihat bahwa dari _Tabel 1. Exploratory Data Analysis (EDA) Variabel_ dataset ini telah di bersihkan dan normalisasi oleh pembuat, sehingga mudah digunakan dan ramah bagi pemula.
* Dataset berupa CSV (Comma-Sepereted Values)
* Dataset memiliki 8000 sample dengan 8 fitur
* Dataset memiliki 7 fitur bertipe float64 dan 2 bertipe object
* Tidak terdapat missing value

### Variabel pada Dataset
* Ukuran - ukuran buah
* Berat - berat buah
* Rasa manis - rasa manis buah
* Kelembutan - kelembutan buah
* Waktu Panen - jumlah waktu yang telah berlalu sejak buah dipanen
* Kematangan - kematangan buah
* Keasaman - keasaman buah
* Kualitas - kualitas buah

Semua fitur memiliki pengaruh kualitas buah pisang

### Exploratory Data Analysis - Univariate Analysis/Distribution of Target Variabel
![image](https://github.com/user-attachments/assets/3a9fb456-e5a8-4e6a-ae5d-c6d26399d2bf)

Gambar 1.1 Analisis Univariate (Data Kategori)

Berdasarkan Gambar 1.1 , dapat dilihat bahwa distribusi data katagorik Quality yang terdiri dari good dan bad kualitas pisang, yang mana nilai data bad terdiri dari 3994 dan good terdiri dari 4006, yang mana menunjukan perbandingan data yang tidak terlalu jauh. 

![image](https://github.com/user-attachments/assets/09d81ab4-1f5e-4d0b-9f5a-4c36acf44609)
Gambar 1.2 Analisis Univariate (Data Numerik)

#### Variabel yang Dianalisis

### 1. Size (Ukuran)
- **Mean**: -0.51
- Ukuran rata-rata buah berkisar antara -8 hingga 8, dengan sebagian besar data berada di antara -2 hingga 2.
- Distribusi ukuran buah mendekati normal, tetapi sedikit miring ke kiri, menunjukkan ukuran buah umumnya sedikit di bawah rata-rata.

### 2. Weight (Berat)
- **Mean**: -0.99
- **Nilai Maksimum**: 3.08
- Distribusi berat berkisar dari -8 hingga 6, dengan nilai rata-rata sedikit di bawah 0. Berat buah mayoritas berkumpul di nilai negatif, dengan beberapa outliers pada nilai maksimum.

### 3. Sweetness (Tingkat Kemanisan)
- **Mean**: -0.48
- Distribusi tingkat kemanisan buah berkisar antara -8 hingga 8, dengan puncak data di sekitar -2 hingga 0. Sebagian besar buah dalam dataset memiliki tingkat kemanisan sedikit di bawah nilai rata-rata yang dinormalisasi.

### 4. Softness (Kerenyahan / Tekstur)
- **Rentang**: 0 hingga 2
- Distribusi tekstur buah lebih berkonsentrasi di sisi positif, menunjukkan bahwa sebagian besar buah lebih renyah.
- Variabel ini memiliki bentuk distribusi yang lebih tajam dibandingkan variabel lainnya.

### 5. Harvest Time (Waktu Panen)
- **Mean**: 0.50
- Distribusi waktu panen berkisar dari -8 hingga 8, dengan puncak data di sekitar nilai netral (0). Ini menunjukkan bahwa sebagian besar buah dipanen pada waktu yang diharapkan berdasarkan distribusi normalisasi.

### 6. Ripeness (Kematangan)
- **Mean**: 0.53
- Distribusi kematangan buah mendekati normal, dengan nilai rata-rata sedikit di atas 0. Ini berarti sebagian besar buah memiliki tingkat kematangan yang optimal.

### 7. Acidity (Keasaman)
- **Mean**: 0.06
- Distribusi keasaman buah menunjukkan nilai yang hampir simetris dengan puncak data di sekitar 0. Sebagian besar buah memiliki tingkat keasaman yang mendekati rata-rata yang dinormalisasi, tanpa variasi ekstrem.

## Kesimpulan
Setelah melihat distribusi dari variabel-variabel di atas, dapat disimpulkan bahwa:
- Distribusi untuk sebagian besar variabel mendekati distribusi normal, dengan beberapa pengecualian seperti **Softness**, yang menunjukkan puncak yang lebih tajam di sisi positif.
- Data numerik lainnya, seperti **Weight**, **Sweetness**, dan **Size**, menunjukkan nilai rata-rata yang sedikit di bawah nol, mengindikasikan distribusi yang sedikit miring ke kiri.

### EDA - Multivariate Analysis
![image](https://github.com/user-attachments/assets/ddac66a4-b3e8-4366-bd05-6070d0ea9460)

Gambar 2.1 Analisis Multivariate

![image](https://github.com/user-attachments/assets/aab8bacf-4b2d-422f-a74c-96559e14602c)

Gambar 2.2 Analisis Matriks Korelasi

### Gambar 2.1 : Analisis Multivariate
- **Pola Acak:** Sebagian besar fitur menunjukkan pola acak, mengindikasikan hubungan yang lemah.
- **Size vs Sweetness:** Korelasi negatif terlihat; semakin kecil ukuran buah, semakin manis rasanya.
- **Softness vs Sweetness:** Tidak ada korelasi jelas, menunjukkan tekstur tidak memengaruhi kemanisan.

### Gambar 2.2: Matriks Korelasi
- **Size dan Harvest Time:** Korelasi positif (**0.580**), semakin besar ukuran buah, semakin lama waktu panen.
- **Weight dan Acidity:** Korelasi positif (**0.430**), buah yang lebih berat cenderung lebih asam.
- **Quality dengan Harvest Time dan Ripeness:** Korelasi masing-masing **0.387** dan **0.361**, menunjukkan pengaruh waktu panen dan kematangan terhadap kualitas.

### Kesimpulan
- **Analisis Multivariate:** Sebagian besar fitur menunjukkan hubungan lemah, kecuali beberapa seperti **Size** dan **Sweetness**.
- **Matriks Korelasi:** Korelasi signifikan ditemukan antara fitur seperti **Size**, **Weight**, dan **Acidity**.

---

## Data Prepation
Pada proses _Data Prepation_ dilakukan kegiatan seperti _Data Gathering, Data Assesing, _


## Referensi
[1] Kementerian Pertanian Republik Indonesia. (2021). Statistik Pertanian 2021. https://www.pertanian.go.id/home/?show=page&act=view&id=61

[2] Bugaud, C., Chillet, M., Beauté, M. P., & Dubois, C. (2006). Physicochemical analysis of mountain bananas from the French West Indies. Scientia Horticulturae, 108(2), 167-172. https://doi.org/10.1016/j.scienta.2006.01.025. Jurnal ini menganalisis sifat fisikokimia pisang, termasuk tekstur dan kematangan.

[3] Marimin, M., Arkeman, Y., Luthfiyanti, R., & Juharni, S. R. (2019). Intelligent Supply Chain Management of Agricultural Products Based on Internet of Things. IOP Conference Series: Earth and Environmental Science, 347(1), 012068. https://doi.org/10.1088/1755-1315/347/1/012068
