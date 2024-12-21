# Machine Learning Pipeline with TFX (IMDB Reviews)

Nama: Umar Sani

Username dicoding: umarsani16

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| Masalah | Klasifikasi review film berdasarkan dua kategori: positif dan negatif |
| Solusi machine learning | Analisis sentimen dengan deep learning |
| Metode pengolahan | Mengubah semua teks ulasan menjadi lowercase dan mengubah label menjadi numerik |
| Arsitektur model | Text Vectorization, Embedding, 2 layer LSTM, dan 2 layer Dropout  |
| Metrik evaluasi | Metrik evaluasi yang digunakan yaitu akurasi, presisi, recall, dan f1-score |
| Performa model | Akurasi: 0.83337, Loss: 0.44146, Precision: 0.88048, Recall: 0.77521, F1-Score: 0.82450 |
| Opsi Deployment | Deployment ke Railway menggunakan Docker |
| Web App | Tautan model: [Model Metadata](https://dicoding-mlops-2-production.up.railway.app/v1/models/imdb-review-model/metadata). Tautan monitoring: [Grafana](https://umarsani1605.grafana.net/public-dashboards/5b13bb92e68c44c88207f3c85053da9e?orgId=1) |
| Monitoring | Terdapat banyak metrik yang dapat divisualisasikan, seperti request count, request latency, dan lain-lain. |