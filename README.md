# GrokkingDeepLearning
Grokking Deep Learning Book
# Grokking Deep Learning — Bab 1–6

|           |                         |
| --------- | ----------------------- |
| **Nama**  | Zacky Yusup Hakim       |
| **NIM**   | 101032300183            |
| **Kelas** | BS1TK-47-REG-G13        |

Repositori ini berisi rangkuman dan implementasi kode Python untuk **Bab 1 sampai Bab 6** dari buku *Grokking Deep Learning* (Andrew W. Trask, Manning, 2019). Setiap bab disajikan dalam satu Jupyter Notebook yang menggabungkan penjelasan konsep (Bahasa Indonesia) dengan contoh kode, visualisasi, dan ringkasan. Seluruh output sudah tersimpan di notebook, sehingga dapat dibaca tanpa dijalankan ulang.

Sesuai filosofi buku, semua neural network dibangun **dari nol hanya dengan NumPy**, mulai dari satu bobot, *gradient descent*, hingga deep neural network pertama dengan *backpropagation*.

Repositori ini dibuat untuk **Tugas 2 (Enrichment for Deep Learning Classes): Code Reproduction + Theoretical Deep-Dive**.

---

## Daftar Isi

1. [Struktur Proyek](#struktur-proyek)
2. [Instalasi & Cara Menjalankan](#instalasi--cara-menjalankan)
3. [Bab 1 — Introducing Deep Learning](#bab-1--introducing-deep-learning)
4. [Bab 2 — Fundamental Concepts](#bab-2--fundamental-concepts)
5. [Bab 3 — Forward Propagation](#bab-3--forward-propagation)
6. [Bab 4 — Gradient Descent](#bab-4--gradient-descent)
7. [Bab 5 — Generalizing Gradient Descent](#bab-5--generalizing-gradient-descent)
8. [Bab 6 — Backpropagation](#bab-6--backpropagation)
9. [Dataset](#dataset)
10. [Library yang Digunakan](#library-yang-digunakan)
11. [Referensi](#referensi)

---

## Struktur Proyek

```
Grokking Deep Learning/
├── README.md
└── notebooks/
    ├── 01_Introducing_Deep_Learning.ipynb
    ├── 02_Fundamental_Concepts.ipynb
    ├── 03_Forward_Propagation.ipynb
    ├── 04_Gradient_Descent.ipynb
    ├── 05_Generalizing_Gradient_Descent.ipynb
    └── 06_Backpropagation.ipynb
```

## Instalasi & Cara Menjalankan

```
pip install numpy matplotlib scikit-learn jupyter
cd notebooks
jupyter notebook
```

Semua bab memakai data mainan yang didefinisikan langsung di notebook, sehingga **tidak perlu mengunduh dataset**. Satu-satunya pengecualian adalah ilustrasi tambahan di Bab 5, yang memakai dataset digit bawaan `scikit-learn` (sudah termasuk saat instalasi).

Sel yang berlabel **"Tambahan"** adalah contoh pelengkap yang tidak ada di kode buku. Bab 1 dan Bab 2 hampir tidak memiliki kode di buku (bab pengantar dan konseptual), sehingga ilustrasi kodenya berlabel "Tambahan".

---

## Bab 1 — Introducing Deep Learning

📓 [`01_Introducing_Deep_Learning.ipynb`](01_Introducing_Deep_Learning.ipynb)

Bab motivasi: **mengapa** deep learning layak dipelajari, **mengapa** buku ini berbeda, dan **apa** yang dibutuhkan untuk mulai.

**Materi:**

1. Selamat datang & arti kata *grok*
2. Tiga alasan belajar deep learning
3. Apakah sulit dipelajari? (*fun payoff*)
4. Mengapa membaca buku ini (analogi NASCAR, palu, dan paku)
5. Yang dibutuhkan: Jupyter, NumPy, matematika SMA, masalah pribadi, Python
6. Pemanasan Python (list, loop, fungsi, `w_sum`)
7. Pemanasan NumPy & benchmark vektorisasi
8. Matematika SMA yang dibutuhkan (perkalian, grafik x–y, slope)
9. Sekilas neural network yang **belajar**
10. Peta perjalanan buku

**Rangkuman:**

- Deep learning adalah alat yang kuat untuk **otomatisasi kecerdasan secara bertahap** dan menyenangkan untuk dipelajari.
- Pendekatan buku: **intuisi + kode dari nol**, dengan matematika setingkat SMA.
- Vektorisasi NumPy memberi hasil yang sama dengan loop Python tetapi jauh lebih cepat (pada benchmark di notebook, ratusan kali lebih cepat).

---

## Bab 2 — Fundamental Concepts

📓 [`02_Fundamental_Concepts.ipynb`](02_Fundamental_Concepts.ipynb)

*How do machines learn?* — memetakan lanskap machine learning, disertai ilustrasi kode kecil untuk tiap konsep.

**Materi:**

1. Deep learning ⊂ machine learning ⊂ AI
2. Machine learning: *monkey see, monkey do*
3. *Supervised learning* (harga saham Senin → Selasa)
4. *Unsupervised learning* (clustering *puppies, pizza, kittens, hot dog, burger*; k-means)
5. *Parametric* vs *nonparametric*
6. *Supervised parametric learning*: predict → compare → learn
7. *Unsupervised parametric learning*
8. *Nonparametric learning* (k-NN)
9. Empat kategori algoritma

**Rangkuman:**

- ***Supervised***: *apa yang diketahui* → *apa yang ingin diketahui*; ***unsupervised***: mengelompokkan data tanpa label.
- ***Parametric***: jumlah parameter tetap (*trial and error*, memutar kenop); ***nonparametric***: parameter ditentukan data (*counting*).
- Deep learning adalah **parametric learning** dengan siklus **predict → compare → learn**.

---

## Bab 3 — Forward Propagation

📓 [`03_Forward_Propagation.ipynb`](03_Forward_Propagation.ipynb)

Langkah pertama siklus belajar: **predict**.

**Materi:**

1. Neural network 1 input → 1 output
2. *Multiple inputs*: *weighted sum* dan kontribusi tiap input
3. *Dot product* sebagai ukuran kemiripan
4. *Multiple outputs*: *elementwise multiplication*
5. Banyak input & banyak output: *vector–matrix multiplication*
6. Predicting on predictions: *hidden layer* (Python murni & NumPy)
7. Primer NumPy (broadcasting, aturan *shape*)

**Rangkuman:**

- ***Forward propagation*** = mengalirkan input melalui bobot untuk menghasilkan prediksi.
- ***Dot product*** mengukur **kemiripan** input dengan bobot; kontribusi = input × bobot.
- Layer bisa **ditumpuk**; perhatikan aturan *shape* `(m,n)·(n,k) → (m,k)`. Versi Python murni dan NumPy menghasilkan prediksi yang sama.

---

## Bab 4 — Gradient Descent

📓 [`04_Gradient_Descent.ipynb`](04_Gradient_Descent.ipynb)

Langkah **compare** dan **learn**: algoritma belajar terpenting dalam deep learning.

**Materi:**

1. *Mean squared error* & mengapa error dikuadratkan
2. *Hot and cold learning* & kelemahannya (*step size* tetap)
3. Arah & besar perubahan: efek *stopping*, *negative reversal*, *scaling*
4. Satu iterasi dan beberapa langkah *gradient descent*
5. Apa sebenarnya `weight_delta`: *derivative* (+ verifikasi numerik)
6. Memakai *derivative* untuk belajar
7. *Overcorrection* & *divergence*
8. *Alpha* (*learning rate*) dan pencarian alpha berdasarkan orde besaran

**Rangkuman:**

- `delta = pred - goal`, `weight_delta = delta * input`, `weight -= alpha * weight_delta`.
- `weight_delta` adalah **derivative** error terhadap bobot (slope kurva error); hasil verifikasi numerik cocok dengan rumus turunan.
- Input besar dapat menyebabkan **divergence** (pada contoh `input = 2`, error meledak); diatasi dengan **alpha**.

---

## Bab 5 — Generalizing Gradient Descent

📓 [`05_Generalizing_Gradient_Descent.ipynb`](05_Generalizing_Gradient_Descent.ipynb)

Gradient descent untuk **banyak bobot sekaligus**.

**Materi:**

1. *Multiple inputs* (satu delta, banyak *weight delta*)
2. Mengamati langkah learning & pentingnya normalisasi input
3. *Freezing one weight*
4. *Multiple outputs*
5. *Multiple inputs & outputs* (*outer product*)
6. Visualisasi bobot sebagai template pada dataset digit (jaringan 64 → 10)

**Rangkuman:**

- Setiap bobot diperbarui dengan `delta_output × input_bobot`; `weight_deltas = outer(delta, input)`.
- Error ditentukan **bersama** oleh semua bobot; meski satu bobot dibekukan, bobot lain tetap membuat error turun.
- Jaringan satu layer pada dataset digit 8×8 mencapai akurasi uji ≈ 95%, dan bobotnya dapat divisualisasikan sebagai "template" tiap angka. (Catatan: kode notebook resmi buku untuk sel pertama bab ini kekurangan satu baris `weight_deltas`, yang sudah diperbaiki di sini.)

---

## Bab 6 — Backpropagation

📓 [`06_Backpropagation.ipynb`](06_Backpropagation.ipynb)

Membangun **deep neural network pertama** pada *streetlight problem*.

**Materi:**

1. *Streetlight problem* & matriks data
2. Belajar dari satu contoh dan dari seluruh dataset (*stochastic*, *full batch*, *batch* GD)
3. *Up and down pressure*: jaringan belajar korelasi
4. *Edge case*: *overfitting* & *conflicting pressure*
5. Korelasi tidak langsung & menciptakan korelasi dengan *hidden layer*
6. *Backpropagation*: *long-distance error attribution*
7. Linear vs nonlinear (ReLU)
8. *Backpropagation* dalam kode & verifikasi dengan gradien numerik
9. Versi *full batch* dan pengaruh jumlah *hidden node*

**Rangkuman:**

- Jika tidak ada korelasi langsung, **hidden layer** dibutuhkan untuk **menciptakan korelasi**.
- Dua layer linear = satu layer linear (error tidak turun ke 0), sehingga dibutuhkan **nonlinearitas** (ReLU); dengan ReLU, error turun mendekati 0.
- **Backpropagation**: `layer_1_delta = layer_2_delta · W_12ᵀ ⊙ relu'(layer_1)`; hasilnya cocok dengan gradien numerik (selisih ~1e-11).

---

## Dataset

| Bab    | Dataset                                                   | Sumber                          |
| ------ | --------------------------------------------------------- | ------------------------------- |
| 1–4, 6 | Data mainan (toes/wlrec/nfans, streetlight, dll.)         | Didefinisikan di notebook       |
| 5      | Data mainan; dataset digit 8×8 (ilustrasi tambahan)       | Notebook / `sklearn.datasets`   |

## Library yang Digunakan

`numpy` · `matplotlib` · `scikit-learn` (hanya untuk dataset digit pada Bab 5)

## Referensi

- Trask, A. W. (2019). *Grokking Deep Learning*. Manning Publications.
- Repositori kode resmi: <https://github.com/iamtrask/Grokking-Deep-Learning>
