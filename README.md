# 📊 Dashboard Analisis Attrition - Jaya Jaya Maju

Dashboard interaktif ini dirancang menggunakan **Streamlit** untuk menganalisis data *employee attrition* di perusahaan Jaya Jaya Maju. Dashboard mencakup berbagai aspek seperti demografi, kepuasan kerja, kompensasi, dan feature importance dari model machine learning.

---

## 📁 Struktur File

```

project-folder/
│
├── dashboard.py                # File utama Streamlit dashboard
├── processed\_data.csv          # Dataset yang sudah diproses
├── feature\_importance.csv      # Output feature importance dari model ML
├── env/                        # Virtual environment (opsional)
└── README.md                   # Petunjuk penggunaan

````

---

## 🚀 Cara Menjalankan Dashboard

### 1. **Buat Virtual Environment (Opsional tetapi Direkomendasikan)**

```bash
python -m venv env
source env/bin/activate        # Linux/Mac
env\Scripts\activate           # Windows
````

### 2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

Jika belum punya `requirements.txt`, berikut isi minimalnya:

```txt
streamlit
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
reportlab
```

### 3. **Jalankan Dashboard di VS Code / Terminal**

```bash
streamlit run dashboard.py
```

---

## 📌 Fitur Utama

1. **Overview Attrition**

   * Total karyawan
   * Rata-rata usia
   * Tingkat attrition

2. **Analisis Demografis**

   * Distribusi usia, jenis kelamin, status pernikahan, dan pendidikan

3. **Analisis Pekerjaan**

   * Attrition per departemen, job role, job level, dan lama bekerja

4. **Faktor Kepuasan**

   * Heatmap korelasi faktor kepuasan kerja dengan attrition

5. **Analisis Kompensasi**

   * Distribusi pendapatan, kenaikan gaji, dan opsi saham

6. **Feature Importance**

   * Visualisasi fitur-fitur yang paling memengaruhi attrition menurut model Random Forest

---

## 🧰 Filter yang Tersedia di Sidebar

* **Departemen**
* **Usia (rentang usia)**
* **Jenis Kelamin**

Filter ini akan memengaruhi seluruh tampilan dan perhitungan pada dashboard.

---

## 📦 Dataset

* **processed\_data.csv**: Dataset karyawan setelah preprocessing
* **feature\_importance.csv**: Hasil evaluasi feature importance dari model

Pastikan kedua file ini tersedia di direktori yang sama dengan `dashboard.py`.

---

## ✨ Contoh Tampilan Dashboard

![screenshot](screenshot_dashboard.png)  <!-- Tambahkan screenshot jika ada -->

---

## 🛠️ Catatan Tambahan

* Jika terjadi error `ValueError: Value of 'y' is not the name of a column`, pastikan nama kolom di file CSV sesuai (`Feature`, bukan `feature`, dll).
* Jalankan `st.write(df.columns)` di Streamlit untuk memeriksa nama kolom.

---

## 👤 Pembuat

> Proyek ini dibuat oleh \[MUCHLISH/surv.massenrempulu@gmail.com/ibnu_tahur86], peserta Digitalent - Laskar AI, 2025.

```

---

### 📎 Tips Tambahan

- Simpan file ini sebagai `README.md` di root proyek.
- Jika kamu menggunakan GitHub, file ini akan otomatis ditampilkan di halaman depan repositori.
- Kalau kamu ingin `requirements.txt` juga saya bantu buatkan, tinggal beri tahu saja.

Perlu ditambahkan logo, badge, atau link ke dataset GitHub juga?
```
