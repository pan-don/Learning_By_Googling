# <h1 align="center">CO2 CARBON EMISSIONS IN VEHICLES</h1>

<h1 align="center">
    <img src="https://t3.ftcdn.net/jpg/08/37/02/14/360_F_837021439_lLG2YvpdyV0AbwgD1Yk1LyL88HdVIQOp.jpg">
</h1>

### Problem Statment:

Kendaraan bermotor merupakan salah satu penyumbang utama emisi CO₂ yang memperburuk pemanasan global dan menurunkan kualitas udara di kota-kota besar. Dengan meningkatnya emisi CO₂ setiap harinya, dampak negatif terhadap lingkungan dan kesehatan manusia semakin terasa. Untuk mengatasi masalah ini, sebuah organisasi lingkungan kota bekerja sama dengan ahli data dalam sebuah proyek untuk mengembangkan model machine learning yang dapat mengidentifikasi polusi CO₂ yang dihasilkan oleh berbagai jenis kendaraan. Data untuk model ini dikumpulkan melalui inspeksi langsung dengan mencatat karakteristik berbagai tipe kendaraan seperti jenis kendaraan, konsumsi bahan bakar, jenis bahan bakar yang digunakan, serta kadar emisi CO₂ yang dihasilkan. Data yang terkumpul membentuk Dataset **co2_pollution.csv**, yang akan menjadi dasar pengembangan model prediktif. Model ini bertujuan untuk mengidentifikasi faktor-faktor utama penyebab emisi dan memproyeksikan tingkat emisi di masa depan. Hasil analisis ini diharapkan dapat menjadi acuan bagi pemerintah kota dalam merancang kebijakan yang efektif untuk mengurangi emisi CO₂ dan memperbaiki kualitas udara di wilayah perkotaan.

---

### Objective:

Tujuan utama dari proyek ini adalah mengembangkan mengembangkan model machine learning yang dapat dengan akurat mengidentifikasi faktor-faktor yang dapat mempengaruhi emisi CO₂ dari kendaraan bermotor dan memproyeksikan tingkat emisi di masa depan.  Hasil model ini akan digunakan sebagai acuan bagi pemerintah kota dalam merancang kebijakan yang efektif untuk mengurangi polusi udara dan meningkatkan kualitas hidup masyarakat perkotaan.

---

### Data Dictionary:

Dataset **co2_polution.csv** memiliki 4938 baris dan 9 kolom yang memuat informasi tentang karakteristik kendaraan serta tingkat emisi CO₂ yang dihasilkannya. Berikut adalah penjelasan setiap fitur:

> - `vehicle_class`: Kategori kendaraan yang menggambarkan jenis dan ukuran kendaraan.
> - `engine_size`: Kapasitas mesin kendaraan dalam liter, yang menunjukkan seberapa besar mesin kendaraan tersebut.
> - `cylinders`: Jumlah silinder dalam mesin kendaraan, yang sering kali berkaitan dengan daya mesin dan konsumsi bahan bakar.
> - `fuel_type`: Jenis bahan bakar yang digunakan.
> - `fuel_cons_city`: Konsumsi bahan bakar dalam kota (L/100 km), yang menunjukkan efisiensi bahan bakar kendaraan saat digunakan di area perkotaan.
> - `fuel_cons_hwy`: Konsumsi bahan bakar di jalan raya (L/100 km), menggambarkan efisiensi bahan bakar di jalan raya.
> - `fuel_cons_comb`: Konsumsi bahan bakar gabungan, yaitu rata-rata konsumsi bahan bakar dalam kondisi perkotaan dan jalan raya.
> - `fuel_cons_comb_mpg`: Konsumsi bahan bakar gabungan dalam satuan mil per galon (mpg), yang sering digunakan sebagai ukuran efisiensi bahan bakar.
> - `co2`: Jumlah emisi CO₂ yang dihasilkan kendaraan (g/km), yang menjadi target utama dalam model prediksi emisi.

---

### Data Problems:

Dataset **co2_polution.csv** menghadapi beberapa masalah pendataan yang perlu ditangani sebelum proses pemodelan machine learning. Di antaranya adalah adanya data yang hilang (missing values) pada beberapa kolom yang dapat mengurangi kualitas data dan menghambat kinerja model. Nilai ekstrem (outliers) pada fitur-fitur seperti ukuran mesin dan konsumsi bahan bakar juga dapat mengganggu pola data yang valid. Selain itu, fitur numerik memiliki skala data berbeda yang berpotensi mempengaruhi model tertentu. Masalah lain termasuk variabel kategorikal pada fitur `vehicle_class` dan `fuel_type` yang tidak dapat langsung diproses oleh model machine learning. Terakhir korelasi fitur (multikollinearitas) antara beberapa kolom yang bisa menyebabkan kesalahan dalam interpretasi model.