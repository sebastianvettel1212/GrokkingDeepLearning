# Grokking Deep Learning — Bab 1–10

|           |                         |
| --------- | ----------------------- |
| **Nama**  | Zacky Yusup Hakim       |
| **NIM**   | 101032300183            |
| **Kelas** | BS1TK-47-REG-G13        |

Repositori ini berisi rangkuman dan implementasi kode Python untuk **Bab 1 sampai Bab 10** dari buku *Grokking Deep Learning* (Andrew W. Trask, Manning, 2019). Setiap bab disajikan dalam satu Jupyter Notebook yang menggabungkan penjelasan konsep (Bahasa Indonesia) dengan contoh kode, visualisasi, dan ringkasan. Seluruh output sudah tersimpan di notebook, sehingga dapat dibaca tanpa dijalankan ulang.

Sesuai filosofi buku, semua neural network dibangun **dari nol hanya dengan NumPy**, mulai dari satu bobot, *gradient descent*, deep neural network pertama dengan *backpropagation*, hingga regularisasi, fungsi aktivasi, dan *convolutional neural network* pada dataset MNIST.

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
9. [Bab 7 — How to Picture Neural Networks](#bab-7--how-to-picture-neural-networks)
10. [Bab 8 — Regularization and Batching](#bab-8--regularization-and-batching)
11. [Bab 9 — Activation Functions](#bab-9--activation-functions)
12. [Bab 10 — Convolutional Neural Networks](#bab-10--convolutional-neural-networks)
13. [Dataset](#dataset)
14. [Library yang Digunakan](#library-yang-digunakan)
15. [Referensi](#referensi)

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
    ├── 06_Backpropagation.ipynb
    ├── 07_How_To_Picture_Neural_Networks.ipynb
    ├── 08_Regularization_and_Batching.ipynb
    ├── 09_Activation_Functions.ipynb
    └── 10_Convolutional_Neural_Networks.ipynb
```

## Instalasi & Cara Menjalankan

```
pip install numpy matplotlib scikit-learn jupyter
cd notebooks
jupyter notebook
```

Bab 1–7 memakai data mainan yang didefinisikan langsung di notebook (Bab 5 juga memakai dataset digit bawaan `scikit-learn` untuk satu ilustrasi). **Bab 8–10 memakai dataset MNIST** yang dimuat oleh fungsi `load_mnist()` di tiap notebook: fungsi ini mencoba `keras.datasets.mnist` lebih dulu (seperti di buku), dan bila Keras tidak tersedia, mengunduh salinan MNIST dari GitHub secara otomatis (diperlukan koneksi internet saat pertama kali). Pada Google Colab, Keras sudah tersedia.

**Perkiraan waktu menjalankan** (CPU biasa): Bab 8 ≈ 6–8 menit, Bab 9 ≈ 4 menit, Bab 10 ≈ 3–4 menit; Bab lainnya hanya beberapa detik.

Setiap bab diakhiri **tanya-jawab konsep** dan **latihan mandiri beserta jawabannya**. Sel yang berlabel **"Tambahan"** adalah contoh pelengkap yang tidak ada di kode buku. Bab 1 dan Bab 2 hampir tidak memiliki kode di buku (bab pengantar dan konseptual), sehingga ilustrasi kodenya berlabel "Tambahan".

---

## Bab 1 — Introducing Deep Learning

📓 [`01_Introducing_Deep_Learning.ipynb`](notebooks/01_Introducing_Deep_Learning.ipynb)

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
11. Tanya-jawab konsep dan latihan mandiri (dengan jawaban)

**Rangkuman:**

- Deep learning adalah alat yang kuat untuk **otomatisasi kecerdasan secara bertahap** dan menyenangkan untuk dipelajari.
- Pendekatan buku: **intuisi + kode dari nol**, dengan matematika setingkat SMA.
- Vektorisasi NumPy memberi hasil yang sama dengan loop Python tetapi jauh lebih cepat (pada benchmark di notebook, ratusan kali lebih cepat).

---

## Bab 2 — Fundamental Concepts

📓 [`02_Fundamental_Concepts.ipynb`](notebooks/02_Fundamental_Concepts.ipynb)

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
9. Pendalaman dengan kode: *supervised parametric* (banyak kenop), *unsupervised parametric* (peluang keanggotaan klaster), *nonparametric* (menghitung)
10. Empat kategori algoritma
11. Tanya-jawab konsep dan latihan mandiri (dengan jawaban)

**Rangkuman:**

- ***Supervised***: *apa yang diketahui* → *apa yang ingin diketahui*; ***unsupervised***: mengelompokkan data tanpa label.
- ***Parametric***: jumlah parameter tetap (*trial and error*, memutar kenop); ***nonparametric***: parameter ditentukan data (*counting*).
- Deep learning adalah **parametric learning** dengan siklus **predict → compare → learn**.

---

## Bab 3 — Forward Propagation

📓 [`03_Forward_Propagation.ipynb`](notebooks/03_Forward_Propagation.ipynb)

Langkah pertama siklus belajar: **predict**.

**Materi:**

1. Neural network 1 input → 1 output
2. *Multiple inputs*: *weighted sum* dan kontribusi tiap input
3. *Dot product* sebagai ukuran kemiripan
4. *Multiple outputs*: *elementwise multiplication*
5. Banyak input & banyak output: *vector–matrix multiplication*
6. Predicting on predictions: *hidden layer* (Python murni & NumPy)
7. Primer NumPy (broadcasting, aturan *shape*)
8. Pendalaman: bobot sebagai sensitivitas, *dot product* sebagai kemiripan (AND/NOT lunak), penelusuran *vector–matrix multiplication*, bukti dua layer linear = satu layer
9. Tanya-jawab konsep dan latihan mandiri (dengan jawaban)

**Rangkuman:**

- ***Forward propagation*** = mengalirkan input melalui bobot untuk menghasilkan prediksi.
- ***Dot product*** mengukur **kemiripan** input dengan bobot; kontribusi = input × bobot.
- Layer bisa **ditumpuk**; perhatikan aturan *shape* `(m,n)·(n,k) → (m,k)`. Versi Python murni dan NumPy menghasilkan prediksi yang sama.

---

## Bab 4 — Gradient Descent

📓 [`04_Gradient_Descent.ipynb`](notebooks/04_Gradient_Descent.ipynb)

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
9. Pendalaman: *tunnel vision*, *box with rods*, slope kurva error, syarat kestabilan `0 < alpha < 2/input²`, bagian *memorizing*
10. Tanya-jawab konsep dan latihan mandiri (dengan jawaban)

**Rangkuman:**

- `delta = pred - goal`, `weight_delta = delta * input`, `weight -= alpha * weight_delta`.
- `weight_delta` adalah **derivative** error terhadap bobot (slope kurva error); hasil verifikasi numerik cocok dengan rumus turunan.
- Input besar dapat menyebabkan **divergence** (pada contoh `input = 2`, error meledak); diatasi dengan **alpha**.

---

## Bab 5 — Generalizing Gradient Descent

📓 [`05_Generalizing_Gradient_Descent.ipynb`](notebooks/05_Generalizing_Gradient_Descent.ipynb)

Gradient descent untuk **banyak bobot sekaligus**.

**Materi:**

1. *Multiple inputs* (satu delta, banyak *weight delta*)
2. Mengamati langkah learning & pentingnya normalisasi input
3. *Freezing one weight*
4. *Multiple outputs*
5. *Multiple inputs & outputs* (*outer product*)
6. Visualisasi bobot sebagai template pada dataset digit (jaringan 64 → 10) dan visualisasi *dot product* per gambar
7. Pendalaman: lintasan bobot, pengaruh skala input terhadap stabilitas, permukaan error 2 bobot
8. Tanya-jawab konsep dan latihan mandiri (dengan jawaban)

**Rangkuman:**

- Setiap bobot diperbarui dengan `delta_output × input_bobot`; `weight_deltas = outer(delta, input)`.
- Error ditentukan **bersama** oleh semua bobot; meski satu bobot dibekukan, bobot lain tetap membuat error turun.
- Jaringan satu layer pada dataset digit 8×8 mencapai akurasi uji ≈ 95%, dan bobotnya dapat divisualisasikan sebagai "template" tiap angka. (Catatan: kode notebook resmi buku untuk sel pertama bab ini kekurangan satu baris `weight_deltas`, yang sudah diperbaiki di sini.)

---

## Bab 6 — Backpropagation

📓 [`06_Backpropagation.ipynb`](notebooks/06_Backpropagation.ipynb)

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
10. Pendalaman: tekanan naik-turun per contoh, *overfitting* dan *conflicting pressure* (didemonstrasikan), korelasi vs bobot terlatih, aktivasi *hidden node*
11. Tanya-jawab konsep dan latihan mandiri (dengan jawaban)

**Rangkuman:**

- Jika tidak ada korelasi langsung, **hidden layer** dibutuhkan untuk **menciptakan korelasi**.
- Dua layer linear = satu layer linear (error tidak turun ke 0), sehingga dibutuhkan **nonlinearitas** (ReLU); dengan ReLU, error turun mendekati 0.
- **Backpropagation**: `layer_1_delta = layer_2_delta · W_12ᵀ ⊙ relu'(layer_1)`; hasilnya cocok dengan gradien numerik (selisih ~1e-11).

---

## Bab 7 — How to Picture Neural Networks

📓 [`07_How_To_Picture_Neural_Networks.ipynb`](notebooks/07_How_To_Picture_Neural_Networks.ipynb)

Bab konseptual tentang **menyederhanakan cara membayangkan jaringan**. Buku tidak menyertakan kode untuk bab ini, sehingga seluruh kode di notebook adalah ilustrasi (berlabel "Tambahan").

**Materi:**

1. Saatnya menyederhanakan
2. Ringkasan korelasi *(correlation summarization)*: lokal dan global, dengan demo "permintaan" layer berikutnya
3. Visualisasi detail (node dan setiap bobot) dan mengapa tidak skalabel
4. Visualisasi sederhana: strip = vektor, kotak = matriks
5. Dimensi bobot ditentukan oleh layer di sisinya; konsep **arsitektur**
6. Melihat jaringan memprediksi (aliran data)
7. Notasi huruf ($l_0$, $W_0$, …) dan persamaan satu baris
8. Menghubungkan gambar, huruf, dan kode; semuanya berdampingan (selisih 0)
9. Pentingnya alat visualisasi
10. Tanya-jawab konsep dan latihan mandiri (dengan jawaban)

**Rangkuman:**

- **Ringkasan korelasi:** *lokal*: tiap set bobot mengorelasikan input-nya dengan permintaan output-nya; *global*: apa yang dipelajari layer awal ditentukan oleh kebutuhan layer berikutnya.
- Ukuran matriks bobot dapat disimpulkan dari ukuran layer (mis. arsitektur 784 → 40 → 10 punya 31.760 bobot).
- Gambar, huruf, dan kode adalah tiga bahasa untuk hal yang sama; hasil hitungannya identik.

---

## Bab 8 — Regularization and Batching

📓 [`08_Regularization_and_Batching.ipynb`](notebooks/08_Regularization_and_Batching.ipynb)

Jaringan menghadapi data nyata (MNIST) dan **menghafal** data latih; bab ini mengajarkan cara mencegahnya.

**Materi:**

1. *Three-layer network* pada MNIST (784 → 40 → 10)
2. Memorisasi vs generalisasi (analogi cetakan)
3. Asal *overfitting* dan pengaruh jumlah data latih
4. *Early stopping* dengan *validation set*
5. *Dropout*: mask acak, penggandaan, dan bukti numerik bahwa sinyal yang diharapkan tetap sama
6. Mengapa dropout bekerja: *ensembling* dan *co-adaptation*
7. Dropout dalam kode dan perbandingan terkontrol
8. *Batch gradient descent* (kecepatan dan pengaruh `alpha`)
9. Tanya-jawab konsep dan latihan mandiri (dengan jawaban)

**Rangkuman:**

- Pada 1.000 data latih, akurasi latih mencapai 100% sedangkan akurasi uji hanya ≈ 70,8%; akurasi uji sempat mencapai puncak ≈ 81,2% (iterasi ke-10) lalu menurun.
- Lebih banyak data memperkecil *overfitting*: akurasi uji ≈ 62,5% (100 data), ≈ 79,3% (1.000), ≈ 87,7% (5.000).
- *Early stopping* berdasarkan validasi memberi akurasi uji ≈ 80,1% vs ≈ 77,1% bila berlatih terus.
- *Dropout* menyempitkan selisih latih-uji (≈ 0,20 → ≈ 0,12) pada eksperimen terkontrol; *mini-batch* ≈ 11× lebih cepat daripada *stochastic*.
- Catatan kode: pada listing batch GD, pembaruan bobot diletakkan sekali per batch (notebook resmi menjorok ke dalam loop `k`), dan dipakai `alpha = 0.1` sesuai teks buku ("20 kali lebih besar") karena `alpha = 0.001` pada listing hanya mencapai ≈ 61% akurasi uji dalam 300 iterasi.

---

## Bab 9 — Activation Functions

📓 [`09_Activation_Functions.ipynb`](notebooks/09_Activation_Functions.ipynb)

Fungsi apa yang sebaiknya dipasang pada keluaran sebuah layer, dan bagaimana memodelkan probabilitas.

**Materi:**

1. Empat ciri fungsi aktivasi yang baik (ilustrasi pelanggar syarat)
2. Aktivasi *hidden layer*: sigmoid, tanh, ReLU (bentuk dan slope)
3. Aktivasi *output layer*: tanpa aktivasi, sigmoid, **softmax**
4. Masalah inti: input yang saling mirip (MSE pada skor mentah vs softmax)
5. Perhitungan softmax langkah demi langkah dan sifat "persaingan" antarkelas
6. Memasang aktivasi pada *backpropagation*: delta × slope
7. Turunan dalam bentuk output dan **cek gradien numerik**
8. Meningkatkan jaringan MNIST (tanh + softmax + dropout + mini-batch)
9. Perbandingan terkontrol kombinasi aktivasi hidden dan output
10. Tanya-jawab konsep dan latihan mandiri (dengan jawaban)

**Rangkuman:**

- Softmax membuat kelas-kelas **bersaing** (peluang positif, jumlah = 1); untuk softmax + cross-entropy, delta output cukup `target − output`.
- Cek gradien numerik: selisih gradien *backprop* vs numerik ≈ 1e-10.
- Jaringan tanh + softmax + dropout + mini-batch mencapai akurasi uji ≈ 87,0% pada 1.000 data latih.
- Pada perbandingan terkontrol, softmax menaikkan akurasi uji untuk tanh (≈ 80,5% → 87,1%) dan sigmoid (≈ 52,5% → 88,8%), tetapi tidak untuk ReLU (≈ 80,6% → 78,9%), sehingga manfaatnya bergantung pada kombinasi aktivasi dan `alpha`.

---

## Bab 10 — Convolutional Neural Networks

📓 [`10_Convolutional_Neural_Networks.ipynb`](notebooks/10_Convolutional_Neural_Networks.ipynb)

Menggunakan ulang bobot yang sama di banyak posisi: **convolutional layer**, dibangun dari nol dengan NumPy.

**Materi:**

1. Menggunakan ulang bobot di banyak tempat dan jumlah parameter
2. *Convolutional layer*: kernel 3×3 yang bergeser, *feature map* (demo kernel tepi buatan tangan)
3. Implementasi NumPy: alur *shape* dan bukti identik dengan `sliding_window_view`
4. Pelatihan CNN pada MNIST (kode buku)
5. Pemeriksaan tanda update kernel (`-=` pada buku vs `+=`)
6. Visualisasi kernel yang dipelajari dan perbandingan dengan jaringan padat
7. Tanya-jawab konsep dan latihan mandiri (dengan jawaban)

**Rangkuman:**

- Layer konvolusi (16 kernel 3×3) hanya memakai **144 bobot**, dibanding ±7,84 juta bobot untuk layer padat dengan jumlah hidden node yang sama (10.000).
- CNN kode buku mencapai akurasi uji ≈ 87,8% setelah 300 iterasi (jaringan padat 100 hidden: ≈ 85,7%).
- Tanda update kernel pada buku (`kernels -= alpha * k_update`) tidak konsisten dengan arah delta (`target − prediksi`). Pada percobaan di notebook, versi `+=` jauh lebih cepat di awal (akurasi uji ≈ 74% vs ≈ 1% pada iterasi ke-20), tetapi akurasi akhir keduanya setara (≈ 87,8% vs ≈ 87,0%; satu *seed*).

---

## Dataset

| Bab    | Dataset                                                   | Sumber                          |
| ------ | --------------------------------------------------------- | ------------------------------- |
| 1–4, 6 | Data mainan (toes/wlrec/nfans, streetlight, dll.)         | Didefinisikan di notebook       |
| 5      | Data mainan; dataset digit 8×8 (ilustrasi tambahan)       | Notebook / `sklearn.datasets`   |
| 7      | Data *streetlight* (dari Bab 6)                           | Didefinisikan di notebook       |
| 8–10   | MNIST (60.000 latih, 10.000 uji; 1.000 gambar pertama dipakai melatih) | `keras.datasets.mnist` atau salinan GitHub (otomatis) |

## Library yang Digunakan

`numpy` · `matplotlib` · `scikit-learn` (hanya untuk dataset digit pada Bab 5) · `keras` (opsional, untuk memuat MNIST; ada cadangan otomatis dari GitHub)

## Referensi

- Trask, A. W. (2019). *Grokking Deep Learning*. Manning Publications.
- Repositori kode resmi: <https://github.com/iamtrask/Grokking-Deep-Learning>
- Salinan MNIST cadangan: <https://github.com/mnielsen/neural-networks-and-deep-learning> (berkas `data/mnist.pkl.gz`)
