![image](https://github.com/user-attachments/assets/821e28a1-d0ab-4d2e-ba44-6b5b2a7120c9)

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
  * XGBRF (XGBoost Random Forest) adalah hybrid antara XGBoost dan Random Forest. Ini menggabungkan teknik bagging dari Random Forest dengan kekuatan XGBoost sebagai base learner [13].

## Data Understanding
| Jenis | Keterangan |
| ------ | ------ |
| Title | _Banana Quality_ |
| Source | [Kaggle](https://www.kaggle.com/datasets/l3llff/banana) |
| Maintainer | [l3LlFF](https://www.kaggle.com/l3llff) |
| License | Other (specified in description) |
| Visibility | Publik |
| Tags | _Earth and Nature, Education, Food, Data Visualization, Exploratory Data Analysis, Binary Classification_ |
| Usability | 10.00 |

Data yang digunakan pada proyek ini adalah _Banana Quality_ yang diunduh dari platform [Kaggle](https://www.kaggle.com/datasets/l3llff/banana). Karakterisitik terdiri dari 8 jenis fitur yang mencakup fitur numerik dan non-numerik (kategorikal). 

Berikut adalah informasi pada dataset. Data yang digunakan dalam pembuatan model merupakan data primer, yang disediakan secara publik di kaggle dengan nama dataset yaitu: _Banana Quality_

#### Variabel-variabel pada Banana Quality
* Ukuran - ukuran buah
* Berat - berat buah
* Rasa manis - rasa manis buah
* Kelembutan - kelembutan buah
* Waktu Panen - jumlah waktu yang telah berlalu sejak buah dipanen
* Kematangan - kematangan buah
* Keasaman - keasaman buah
* Kualitas - kualitas buah

#### Berikut adalah tahapan untuk memahami sebuah data:
* Data Loading
* Exploratory Data Analysis - Deskripsi Variabel
* Exploratory Data Analysis - Mengidentifisikan Missing Value, Outlier, dan hapus fitur yang tidak diperlukan
* Exploratory Data Analysis - Univariate Analysis
* Exploratory Data Analaysis - Multivariate Analysis

#### Data Loading
Bagian ini, dataset secara langsung dibaca dari folder yang sudah di download melalui _Banana Quality_. Dataset yang digunakan adalah _banana_quality.csv_ yang berisikan data-data yang digunakan untuk pelatihan model.
![image](https://github.com/user-attachments/assets/26b3ed9a-c1e7-46c8-a7eb-f81971f5715a)

Gambar 1.1 Data dari _Banana Quality_

Berdasarkan Gambar 1.1 informasi yang didapat dataset sebagai berikut:
* Dataset berupa CSV (Comma-Sepereted Values)
* Dataset memiliki 8000 sample dengan 8 fitur
* Dataset memiliki 7 fitur bertipe float64 dan 2 bertipe object
* Tidak terdapat missing value

#### Exploratory Data Analysis - Deskripsi Variabel
Exploratory Data Analysis (EDA) adalah proses awal dalam menganalisis data untuk memahami karakteristik, pola, anomali, dan memverifikasi asumsi dalam dataset. EDA berfokus pada eksplorasi deskriptif variabel untuk mendapatkan informasi yang lebih mendalam dan memvalidasi integritas data.

Dalam kasus yang berasal dari [Kaggle](https://www.kaggle.com/datasets/l3llff/banana), terdapat sekitar 8 variabel yang berkaitan dengan kualitas pisang, di mana detail lebih lanjut tentang variabel-variabel tersebut tercantum dalam pada laman kaggle tersendiri. EDA berfungsi sebagai langkah awal untuk memahami keseluruhan dataset sebelum melanjutkan ke analisis yang lebih mendalam atau pemodelan prediktif. Setelah melakukan pengecekkan pada dataset, tidak terdapat fitur/kolom yang memiliki nilai null/NaN. Selanjutnya, karena semua kolom telah memiliki tipe data yang sesuai dilakukan proses pengecekan deskripsi statistik data menggunakan fitur describe(). 

Fungsi describe() sendiri memberikan informasi statistik pada masing-masing fitur/kolom, diantara lain:
* Count adalah jumlah sampel pada data.
* Mean adalah nilai rata-rata.
* Std adalah standar deviasi.
* Min yaitu nilai minimum setiap kolom.
* 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
* 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
* 75% adalah kuartil ketiga.
* Max adalah nilai maksimum.

#### Exploratory Data Analysis - Mengidentifisikan Missing Value, Outlier, dan hapus fitur yang tidak diperlukan
Pada proses EDA dilakukan kegiatan seperti Data Gathering, Data Assessing, dan Data Cleaning. Pada proses Data Gathering, data diimpor sedemikian rupa agar bisa dibaca dengan baik menggunakan dataframe Pandas. Untuk proses Data Assessing, berikut adalah beberapa pengecekan yang dilakukan:
* Duplicate data (data yang serupa dengan data lainnya).
* Missing value (data atau informasi yang "hilang" atau tidak tersedia)
* Outlier (data yang menyimpang dari rata-rata sekumpulan data yang ada).

Pada proses Data Cleaning yang dilakukan adalah seperti:
* Converting Column Type (Mengubah tipe suatu kolom).
* Train Test Split (membagi data menjadi data latih dan data uji).
* Normalization (mentransformasi data ke dalam skala yang seragam sehingga semua fitur atau atribut memiliki rentang nilai yang sebanding).

Pada projek kasus ini tidak ditemukannya data duplikat dan _missing value_. Adapun untuk outlier juga dilakukan dengan metode _dropping_ menggunakan metode IQR. IQR dihitung dengan mengurangkan kuartil ketiga (Q3) dari kuartil pertama (Q1) sebagaimana rumusnya berikut:

$$IQR = Q_3 - Q_1$$

* Q1 sebagai kuartil pertama
* Q3 sebagai kuartil ketiga
Setelah menggunakan metode IQR untuk menghilangkan _outlier_ pada dataset jumlah dataset menjadi 7657 yang awalnya adalah 8000. Pada proyek ini digunakan _Train Test Split_ pada library _sklearn.model_selection_ untuk membagi dataset menjadi data latih dan data uji dengan pembagian sebesar 30:70 dan random state sebesar 42. Pada projek kasus ini digunakan _Normalization_ pada libarary _sklearn.preprocessing_ untuk menormalisasikan dataset. Semua proses ini diperlukan dalam rangka membuat model yang baik

### Exploratory Data Analysis - Univariate Analysis/Distribution of Target Variabel
![image](https://github.com/user-attachments/assets/3a9fb456-e5a8-4e6a-ae5d-c6d26399d2bf)

Gambar 2.1 Analisis Univariate (Data Kategori)

Berdasarkan Gambar 2.1 , dapat dilihat bahwa distribusi data katagorik Quality yang terdiri dari good dan bad kualitas pisang, yang mana nilai data bad terdiri dari 3994 dan good terdiri dari 4006, yang mana menunjukan perbandingan data yang tidak terlalu jauh. 

![image](https://github.com/user-attachments/assets/09d81ab4-1f5e-4d0b-9f5a-4c36acf44609)

Gambar 2.2 Analisis Univariate (Data Numerik)

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

Gambar 3.1 Analisis Multivariate

![image](https://github.com/user-attachments/assets/aab8bacf-4b2d-422f-a74c-96559e14602c)

Gambar 3.2 Analisis Matriks Korelasi

### Analisis Multivariate
- **Pola Acak:** Sebagian besar fitur menunjukkan pola acak, mengindikasikan hubungan yang lemah.
- **Size vs Sweetness:** Korelasi negatif terlihat; semakin kecil ukuran buah, semakin manis rasanya.
- **Softness vs Sweetness:** Tidak ada korelasi jelas, menunjukkan tekstur tidak memengaruhi kemanisan.

### Matriks Korelasi
- **Size dan Harvest Time:** Korelasi positif (**0.580**), semakin besar ukuran buah, semakin lama waktu panen.
- **Weight dan Acidity:** Korelasi positif (**0.430**), buah yang lebih berat cenderung lebih asam.
- **Quality dengan Harvest Time dan Ripeness:** Korelasi masing-masing **0.387** dan **0.361**, menunjukkan pengaruh waktu panen dan kematangan terhadap kualitas.

### Kesimpulan
- **Analisis Multivariate:** Sebagian besar fitur menunjukkan hubungan lemah, kecuali beberapa seperti **Size** dan **Sweetness**.
- **Matriks Korelasi:** Korelasi signifikan ditemukan antara fitur seperti **Size**, **Weight**, dan **Acidity**.

---

## Data Prepation
Pada bagian ini, memiliki 3 tahap persiapan data, diantaranya:
* Encoding fitur kategori
* Pembagian dataset dengan fungsi train_set_split dari library sklearn.
* Standarisasi

#### Encoding Fitur Kategori
Proses encoding fitur kategori menggunakan teknik one-hot-encoding. Teknik ini adalah salah satu metode dalam proses encoding fitur (feature encoding) pada data kategorikal. Tujuannya adalah untuk mengubah variabel kategorikal menjadi representasi biner yang dapat digunakan dalam algoritma pembelajaran mesin.

![image](https://github.com/user-attachments/assets/3c5216be-2116-4e4d-95c8-bee3ee6e57ca)

Gambar 4.1 Dataset hasil dari Encoding Fitur Kategori

#### Train-Test-Split
Selanjutnya ialah membagi dataset data latih (train) dan data uji (test). Proses pembagian dataset ini menggunakan library sklearn yaitu train-test-split. Proporsi pembagian adalah 70:30. Hasil pembagian ini menghasilkan sampel untuk train dataset sebesar 5359 dan sampel untuk test dataset sebesar 2298 dari total keseluruhan dataset yaitu sebesar 7657 buah sampel

#### Standarisasi/Normalisasi
Hal ini mungkin tidak dapat dilakukan karena nilai variansnya tidak tinggi

# Modeling
### Algoritma Pemodelan dan Parameter yang Digunakan
Algoritma pada proyek ini melakukan pemodelan dengan 11 algoritma, yaitu:

**1. Random Forest Classifier**

**Cara kerja:**
Random Forest adalah metode ensemble yang membangun banyak pohon keputusan dari subset data yang dipilih secara acak (bootstrap sampling). Pada setiap node dalam pohon, hanya sebagian kecil fitur yang dipertimbangkan untuk melakukan split. Prediksi akhir diambil berdasarkan suara mayoritas dari semua pohon.

**Keuntungan:**
- Kinerja sangat baik pada dataset yang bervariasi.
- Menangani fitur yang tidak relevan dengan baik melalui pemilihan fitur acak.
- Tahan terhadap overfitting karena penggabungan beberapa pohon.

**Kekurangan:**
- Komputasi intensif karena membangun banyak pohon.
- Kurang interpretable karena sulit memahami logika keseluruhan model ensemble.

**2. Logistic Regression**

**Cara kerja:**
Logistic Regression bekerja dengan menghitung log-odds dari variabel target (kelas) sebagai kombinasi linear dari variabel independen (fitur). Model ini menggunakan fungsi logistik (sigmoid) untuk memetakan output ke probabilitas antara 0 dan 1. Prediksi dihasilkan dengan menetapkan threshold (biasanya 0.5) untuk menentukan kelas.

**Keuntungan:**
- Model sederhana, mudah diimplementasikan, dan mudah diinterpretasi.
- Cepat dan efisien, terutama pada dataset yang dapat dipisahkan secara linear.
- Memberikan probabilitas sebagai output, bermanfaat untuk analisis risiko.

**Kekurangan:**
- Mengasumsikan linearitas antara fitur dan log-odds.
- Kurang efektif untuk hubungan non-linear tanpa penambahan fitur atau transformasi.

**3. Support Vector Classifier (SVC)**

**Cara kerja:**
SVC bekerja dengan mencari hyperplane optimal yang memisahkan dua kelas dalam ruang fitur dengan margin maksimum. Algoritma ini mencoba menemukan hyperplane yang memaksimalkan jarak terdekat antara kelas yang berbeda. Jika data tidak linear, kernel trick digunakan untuk mentransformasi data ke dimensi yang lebih tinggi.

**Keuntungan:**
- Efektif dalam ruang dimensi tinggi.
- Fleksibel karena bisa menggunakan berbagai kernel (linear, polynomial, radial).
- Kuat terhadap overfitting dalam ruang dimensi tinggi dengan margin yang baik.

**Kekurangan:**
- Lambat untuk dataset besar, karena membutuhkan banyak komputasi.
- Pemilihan kernel dan parameter yang tepat bisa menjadi tantangan.

**4. MLP Classifier (Multilayer Perceptron)**

**Cara kerja:**
MLP adalah jaringan saraf tiruan feedforward yang terdiri dari lapisan input, lapisan tersembunyi, dan lapisan output. Setiap node (neuron) dihubungkan ke semua neuron di lapisan berikutnya, dan bobot antara neuron diperbarui melalui backpropagation menggunakan algoritma optimasi seperti stochastic gradient descent (SGD).

**Keuntungan:**
- Dapat mempelajari hubungan non-linear yang kompleks antara fitur.
- Kemampuan generalisasi yang baik jika disetel dengan benar.
- Dapat menangani data yang memiliki banyak fitur atau dimensi.

**Kekurangan:**
- Rentan terhadap overfitting, terutama pada dataset kecil.
- Membutuhkan banyak tuning hyperparameter (seperti jumlah neuron, lapisan, learning rate).
- Komputasi yang mahal dan sensitif terhadap scaling fitur.

**5. CatBoost Classifier**

**Cara kerja:**
CatBoost adalah algoritma gradient boosting yang dirancang untuk menangani variabel kategorikal secara otomatis tanpa pre-processing. CatBoost membangun model secara bertahap, di mana setiap model baru berfokus pada memperbaiki kesalahan dari model sebelumnya.

**Keuntungan:**
- Efisien dalam menangani fitur kategorikal tanpa perlu melakukan encoding manual.
- Kinerja baik bahkan tanpa tuning hyperparameter yang ekstensif.
- Mendukung penggunaan GPU untuk mempercepat pelatihan.

**Kekurangan:**
- Memerlukan memori besar untuk dataset besar.
- Mungkin lebih lambat dalam beberapa kasus dibandingkan algoritma boosting lainnya.

**6. AdaBoost Classifier**

**Cara kerja:**
AdaBoost menggabungkan beberapa weak learners (biasanya pohon keputusan sederhana) menjadi satu strong learner. Algoritma ini memberikan bobot lebih besar pada sampel yang salah diklasifikasikan pada iterasi sebelumnya, sehingga model baru lebih fokus pada kesalahan sebelumnya.

**Keuntungan:**
- Sederhana dan mudah diimplementasikan.
- Dapat digunakan dengan berbagai base learners.
- Cenderung tidak overfitting karena fokus pada sampel yang salah klasifikasi.

**Kekurangan:**
- Sensitif terhadap noise dan outlier karena fokus berlebih pada kesalahan.
- Komputasi intensif untuk dataset besar.

**7. Extra Trees Classifier**

**Cara kerja:**
Extra Trees (Extremely Randomized Trees) mirip dengan Random Forest tetapi melakukan split pohon secara acak untuk meningkatkan kecepatan. Algoritma ini memilih split point dan fitur secara acak, menghasilkan variasi yang lebih besar dalam pohon tetapi dengan proses yang lebih cepat.

**Keuntungan:**
- Lebih cepat daripada Random Forest karena split acak.
- Mengurangi varians lebih baik pada data yang berisik.
- Baik untuk dataset besar dan kompleks.

**Kekurangan:**
- Bisa kurang akurat dibandingkan Random Forest karena split yang terlalu acak.
- Pohon yang terlalu dalam dapat terbentuk jika tidak ada batasan.

**8. Gradient Boosting Classifier**

**Cara kerja:**
Gradient Boosting bekerja dengan membangun model secara bertahap, di mana setiap model baru berfokus pada memperbaiki kesalahan yang dibuat oleh model sebelumnya. Algoritma ini menghitung gradien dari fungsi loss dan menggunakan gradien tersebut untuk memperbarui model.

**Keuntungan:**
- Kinerja yang sangat baik pada berbagai jenis dataset.
- Dapat menangani interaksi fitur yang kompleks.
- Fleksibel, dapat dioptimalkan untuk berbagai fungsi loss.

**Kekurangan:**
- Rentan terhadap overfitting jika tidak dikendalikan dengan hati-hati.
- Komputasi mahal untuk dataset besar.

**9. Hist Gradient Boosting Classifier**

**Cara kerja:**
Hist Gradient Boosting adalah varian dari Gradient Boosting yang lebih cepat karena menggunakan binning untuk mengelompokkan nilai fitur ke dalam interval diskret sebelum melakukan split pohon.

**Keuntungan:**
- Lebih cepat daripada Gradient Boosting tradisional.
- Dapat menangani dataset besar secara efisien.
- Kinerja sebanding atau lebih baik daripada Gradient Boosting standar.

**Kekurangan:**
- Kurang akurat pada dataset kecil.
- Kehilangan presisi karena binning fitur, terutama pada data yang kontinu.

**10. XGBoost Classifier**

**Cara kerja:**
XGBoost adalah implementasi gradient boosting yang dioptimalkan untuk efisiensi dan performa. Algoritma ini menggunakan teknik shrinkage (regularization) untuk mencegah overfitting dan mendukung penanganan data yang hilang.

**Keuntungan:**
- Sangat efisien dan cepat, bahkan pada dataset besar.
- Dapat menangani missing values.
- Kinerja sangat baik pada masalah klasifikasi yang kompleks.

**Kekurangan:**
- Membutuhkan banyak tuning hyperparameter untuk hasil optimal.
- Komputasi mahal untuk dataset yang sangat besar.

**11. XGBRF Classifier**

**Cara kerja:**
XGBRF adalah kombinasi dari XGBoost dan Random Forest. Model ini menggunakan pendekatan random forest tetapi dalam konteks boosting, menggabungkan kekuatan dari kedua algoritma untuk meningkatkan stabilitas dan mengurangi overfitting.

**Keuntungan:**
- Menggabungkan kekuatan XGBoost dan Random Forest, memberikan kinerja yang lebih stabil.
- Potensi lebih baik dalam menangani overfitting dibandingkan XGBoost standar.
- Lebih cocok untuk dataset yang beragam.

**Kekurangan:**
- Komputasi lebih mahal karena menggabungkan dua pendekatan.
- Membutuhkan memori yang lebih besar.

#### Tahapan Umum dalam Pemodelan

1. **Pengumpulan Data:**
   Data diambil dari sebuah website data terkenal diseluruh yaitu Kaggle dengan judul _Banana Quality_ dengan jumlah data 80000

2. **Preprocessing Data:**
   Tahapan ini melibatkan persiapan data sebelum digunakan untuk pelatihan model. Kegiatan dalam tahap ini meliputi:
   - **Pembersihan data** (mengatasi missing values, outliers, dsb.)
   - **Transformasi data** (normalisasi atau standardisasi fitur)
   - **Feature engineering** (membuat fitur baru berdasarkan pemahaman domain)
   - **Pembagian data** menjadi data latih (training set) dan data uji (test set), dengan rasio umum 70:30.
     
3. **Pemilihan Algoritma:**
   Berdasarkan karakteristik data dan tujuan pemodelan, algoritma yang sesuai dipilih dari 11 algoritma yang telah dijelaskan.

4. **Pelatihan Model (Training):**
   Pada tahap ini, algoritma yang dipilih dilatih menggunakan data latih. Algoritma akan menyesuaikan parameter internal (misalnya bobot dalam regresi logistik atau node dalam decision tree) untuk menemukan pola dari data.

5. **Evaluasi Model:**
   Setelah model dilatih, model diuji menggunakan data uji untuk mengukur performa model. Evaluasi ini biasanya menggunakan metrik seperti:
   - **Accuracy**: Persentase prediksi yang benar.

6. **Tuning Hyperparameter:**
   Algoritma yang dipilih biasanya memerlukan penyesuaian hyperparameter agar kinerjanya optimal. Ini bisa dilakukan melalui:
   - **Grid Search**: Mencoba semua kombinasi hyperparameter dalam grid.
   - **Random Search**: Memilih secara acak beberapa kombinasi hyperparameter.
   - **Cross Validation**: Teknik yang digunakan untuk mencegah overfitting, dengan memecah data latih ke dalam beberapa subset.

7. **Model Deployment:**
   Setelah model dioptimalkan, tahap terakhir adalah menerapkan model ke dalam sistem produksi agar bisa digunakan untuk memprediksi data baru.

### Parameter Penting yang Digunakan

Berikut adalah parameter utama dari masing-masing algoritma yang digunakan:

### 1. Random Forest Classifier
- `n_estimators`: Jumlah pohon yang akan dibangun di dalam hutan.
- `max_depth`: Kedalaman maksimum setiap pohon keputusan.
- `min_samples_split`: Minimum jumlah sampel yang dibutuhkan untuk membagi node.
- `max_features`: Jumlah maksimum fitur yang dipertimbangkan untuk membagi node.
- `bootstrap`: Jika True, sampel bootstrap digunakan saat membangun pohon.

### 2. Logistic Regression
- `C`: Parameter regulasi untuk mengontrol regularisasi. Nilai kecil memberikan regularisasi yang kuat.
- `solver`: Algoritma yang digunakan untuk mengoptimalkan fungsi loss (misal, ‘liblinear’, ‘lbfgs’).
- `max_iter`: Jumlah iterasi maksimum untuk algoritma optimasi.

### 3. Support Vector Classifier (SVC)
- `C`: Parameter regularisasi. Semakin besar nilainya, semakin ketat aturan dalam memisahkan data.
- `kernel`: Jenis kernel yang digunakan (misal, ‘linear’, ‘poly’, ‘rbf’).
- `gamma`: Parameter kernel ‘rbf’, yang mengontrol jarak pengaruh satu sampel training.

### 4. MLP Classifier (Multilayer Perceptron)
- `hidden_layer_sizes`: Ukuran dan jumlah lapisan tersembunyi dalam jaringan saraf.
- `activation`: Fungsi aktivasi (misal, ‘relu’, ‘tanh’, ‘logistic’).
- `alpha`: Parameter regulasi L2 untuk mencegah overfitting.
- `learning_rate`: Kecepatan pembaruan bobot selama pelatihan.

### 5. CatBoost Classifier
- `iterations`: Jumlah iterasi boosting.
- `depth`: Kedalaman maksimum pohon yang digunakan.
- `learning_rate`: Kecepatan pembaruan model.
- `l2_leaf_reg`: Regularisasi untuk mencegah overfitting.
- `border_count`: Jumlah split pada fitur kontinu.

### 6. AdaBoost Classifier
- `n_estimators`: Jumlah weak learners yang digunakan.
- `learning_rate`: Mengontrol kontribusi setiap weak learner.
- `algorithm`: Tipe boosting yang digunakan (‘SAMME’, ‘SAMME.R’).

### 7. Extra Trees Classifier
- `n_estimators`: Jumlah pohon dalam hutan.
- `max_features`: Jumlah fitur yang dipertimbangkan untuk split setiap node.
- `min_samples_split`: Minimum sampel untuk membagi node.

### 8. Gradient Boosting Classifier
- `n_estimators`: Jumlah boosting stages.
- `learning_rate`: Faktor pengurangan kontribusi setiap tree.
- `max_depth`: Kedalaman maksimum pohon.

### 9. Hist Gradient Boosting Classifier
- `max_iter`: Jumlah boosting iterations.
- `learning_rate`: Faktor yang mengontrol penambahan tree baru.
- `max_leaf_nodes`: Jumlah maksimum node dalam setiap pohon.

### 10. XGBoost Classifier
- `n_estimators`: Jumlah boosting rounds.
- `learning_rate`: Learning rate untuk shrinkage.
- `max_depth`: Kedalaman maksimum setiap pohon.
- `subsample`: Persentase sampel yang digunakan untuk membangun setiap pohon.
- `colsample_bytree`: Persentase fitur yang dipilih secara acak untuk setiap pohon.

### 11. XGBRF Classifier
- `n_estimators`: Jumlah boosting rounds.
- `learning_rate`: Faktor pengurangan kontribusi setiap tree.
- `max_depth`: Kedalaman maksimum pohon.

# Evaluation
Dalam tahap evaluasi, metrik yang digunakan adalah _accuracy_. Accuracy didapatkan dengan menghitung persentase dari jumlah prediksi benar dibagi dengan jumlah seluruh prediksi. Rumusnya sebagai berikut:

$$\text{Accuracy} = \frac{\text{TP + TN}}{\text{TN + TP + FN + FP}} \times 100\%$$

# Penjelasan
- **TP (True Positive):** Jumlah data yang sebenarnya positif dan diprediksi dengan benar sebagai positif.
- **TN (True Negative):** Jumlah data yang sebenarnya negatif dan diprediksi dengan benar sebagai negatif.
- **FP (False Positive):** Jumlah data negatif yang diprediksi secara salah sebagai positif (dikenal juga sebagai Kesalahan Tipe I).
- **FN (False Negative):** Jumlah data positif yang diprediksi secara salah sebagai negatif (dikenal juga sebagai Kesalahan Tipe II).

Rumus ini menunjukkan rasio antara jumlah data yang diklasifikasikan dengan benar (True Positives dan True Negatives) terhadap total data, lalu dikonversi menjadi persentase dengan mengalikannya dengan 100.

Berikut adalah hasil _accuracy_ 11 buah model yang dilatih:
| Model | Accuracy |
| ------ | ------ |
| Random Forest Classifier | 0.9756 |
| Logistic Regression   | 0.8773 |
| SVC  | 0.9830 |
| MLP Classifier  | 0.9804 |
| Cat Boost Classifier  | 0.9809 |
| AdaBoost Classifier  | 0.8908 |
| Extra Trees Classifier    | 0.9822 |
| Gradient Boosting Classifier   | 0.9643 |
| Hist Gradient Boosting Classifier | 0.9756 |
| XGB Classifier   | 0.9752 |
| XGBRF Classifier   | 0.9504 |

Tabel 3.1 Hasil Akurasi

![image](https://github.com/user-attachments/assets/f8237748-8a5a-48c0-a3b8-4effe7eec70d)

Gambar 3.1 Visualisasi Akurasi Model

Berdasarkan **Tabel Hasil Accuracy**, dapat diketahui bahwa model dengan algoritma **SVC** memiliki akurasi tertinggi, yaitu **0.9830**. Oleh karena itu, model **SVC** dipilih untuk digunakan dalam memprediksi kualitas apel. 

#### Pemanfaatan dan Implementasi Model Machine Learning untuk Mendukung Petani dan Distributor

Model machine learning, seperti **Support Vector Classifier (SVC)** yang memiliki akurasi tertinggi sebesar 98.30%, dapat digunakan untuk mendukung petani dan distributor dalam meningkatkan kualitas dan harga jual pisang melalui beberapa cara yang lebih konkret. Berikut adalah beberapa contoh implementasi:

#### 1. Klasifikasi Kualitas Pisang Berdasarkan Data Sensorik
Dengan menggunakan data visualisasi dan sensorik yang dimasukkan ke dalam model SVC, petani dan distributor dapat secara otomatis mengklasifikasikan kualitas pisang dengan cepat dan akurat. Ini mencakup parameter seperti ukuran, warna, dan faktor lainnya yang relevan dengan standar kualitas.

**Contoh Implementasi:**
- Pisang yang baru dipanen dapat langsung dipindai menggunakan perangkat sensor yang terintegrasi dengan model machine learning. Berdasarkan hasil pemindaian, pisang akan langsung diberi label kualitas tinggi, sedang, atau rendah.
- Proses ini menghemat waktu dan tenaga manusia dalam melakukan inspeksi manual, dan hasilnya lebih objektif serta konsisten.

#### 2. Pengurangan Penyortiran Manual dan Human Error
Sebelum implementasi teknologi machine learning, penyortiran pisang biasanya dilakukan secara manual oleh pekerja. Proses ini rentan terhadap kesalahan manusia dan ketidakakuratan. Dengan model machine learning, penyortiran bisa dilakukan secara otomatis berdasarkan fitur yang sudah diajarkan kepada model.

**Contoh Implementasi:**
- Pisang yang disortir secara otomatis berdasarkan kualitasnya dapat langsung dipisahkan dalam berbagai kategori yang disesuaikan dengan kebutuhan pasar, seperti pisang untuk pasar premium atau untuk pengolahan industri (misalnya, bahan baku makanan olahan).
- Proses otomatisasi ini dapat mengurangi jumlah pisang yang salah klasifikasi, meminimalkan kerugian dari produk yang salah dikirimkan, serta meningkatkan kepuasan pelanggan karena konsistensi kualitas.

#### 3. Pengoptimalan Harga Berdasarkan Kualitas
Dengan adanya klasifikasi kualitas yang lebih presisi dan data yang terperinci mengenai kualitas setiap kelompok pisang, distributor dapat menyesuaikan harga jual berdasarkan kualitas secara lebih adil dan akurat. Pisang berkualitas tinggi dapat dijual dengan harga yang lebih tinggi, sementara pisang dengan kualitas sedang atau rendah dapat dijual ke pasar lain, seperti pasar industri.

**Contoh Implementasi:**
- Distributor dapat menggunakan data yang dihasilkan oleh model machine learning untuk menentukan segmentasi pasar yang lebih baik. Pisang dengan kualitas premium dijual ke pasar premium (supermarket atau ekspor), sedangkan pisang dengan kualitas sedang dapat diarahkan ke pasar lokal atau industri pengolahan.
- Model ini juga memungkinkan penetapan harga dinamis berdasarkan permintaan pasar dan prediksi ketersediaan kualitas, sehingga distributor dapat memaksimalkan keuntungan dengan penetapan harga yang lebih optimal.

#### 4. Dukungan Pengambilan Keputusan bagi Petani
Model machine learning yang terintegrasi dengan data dari lapangan dan sensor dapat memberikan wawasan tambahan bagi petani dalam pengambilan keputusan terkait proses budidaya. Misalnya, petani dapat memonitor kondisi tanaman dan kualitas hasil panen secara lebih baik, memungkinkan mereka membuat keputusan yang lebih tepat tentang kapan waktu terbaik untuk memupuk, menyiram, atau memanen.

**Contoh Implementasi:**
- Sistem prediksi berbasis model machine learning dapat memberikan saran terkait perawatan lahan secara personalisasi berdasarkan kondisi real-time tanaman pisang. Hal ini membantu petani dalam menjaga kesehatan tanaman dan meningkatkan produktivitas dengan mengurangi risiko gagal panen akibat faktor lingkungan yang tidak terduga.
- Dengan demikian, kualitas panen dapat terjaga lebih baik, mendukung peningkatan kualitas dan kuantitas hasil pertanian secara keseluruhan.

Dengan demikian, **model SVC** tidak hanya memberikan prediksi yang akurat, tetapi juga memungkinkan adanya efisiensi yang lebih baik dalam berbagai aspek dari **rantai pasokan**, **peningkatan kualitas produk**, dan **pengambilan keputusan**. Implementasi model ini bisa memberikan dampak positif secara langsung bagi petani dan distributor dalam meningkatkan kualitas dan harga jual pisang, meningkatkan daya saing produk, serta mengurangi pemborosan dalam proses distribusi dan penjualan.


## Referensi
1. Kementerian Pertanian Republik Indonesia. (2021). Statistik Pertanian 2021. https://www.pertanian.go.id/home/?show=page&act=view&id=61
2. Bugaud, C., Chillet, M., Beauté, M. P., & Dubois, C. (2006). Physicochemical analysis of mountain bananas from the French West Indies. Scientia Horticulturae, 108(2), 167-172. https://doi.org/10.1016/j.scienta.2006.01.025. Jurnal ini menganalisis sifat fisikokimia pisang, termasuk tekstur dan kematangan.
3. Marimin, M., Arkeman, Y., Luthfiyanti, R., & Juharni, S. R. (2019). Intelligent Supply Chain Management of Agricultural Products Based on Internet of Things. IOP Conference Series: Earth and Environmental Science, 347(1), 012068. https://doi.org/10.1088/1755-1315/347/1/012068
4. Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
5. Cox, D. R. (1958). The regression analysis of binary sequences. Journal of the Royal Statistical Society: Series B (Methodological), 20(2), 215-232.
6. Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine learning, 20(3), 273-297.
7. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533-536.
8. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. Advances in neural information processing systems, 31.
9. Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences, 55(1), 119-139.
10. Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. Machine learning, 63(1), 3-42.
11. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of statistics, 1189-1232.
12. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in neural information processing systems, 30.
13. Chen, T., & Guestrin, C. (2016). Xgboost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, 785-794.
