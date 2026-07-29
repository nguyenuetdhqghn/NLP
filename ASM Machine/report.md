# Báo cáo so sánh mô hình Machine Translation English-Vietnamese
Họ và tên: Trần Hoàng Nguyên
Mã sinh viên: 24022421

## 1. Giới thiệu

Báo cáo này trình bày kết quả thực nghiệm tái hiện và so sánh hai hướng tiếp cận Neural Machine Translation English-Vietnamese trên cùng bộ dữ liệu IWSLT 2015:

- **Case A**: Transformer (PyTorch, from-scratch)
- **Case B**: Google NMT / GNMT (TensorFlow, LSTM + Attention)

## 2. Dữ liệu

- **Bộ dữ liệu**: IWSLT 2015 English-Vietnamese TED Talks
- **Tập train**: ~133K câu song ngữ
- **Tập dev**: tst2012
- **Tập test**: tst2013
- **Nguồn**: https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/

## 3. Kiến trúc mô hình

### 3.1 Case A - Transformer

| Thành phần | Chi tiết |
|------------|----------|
| **Encoder** | 6 lớp Encoder Layer |
| **Decoder** | 6 lớp Decoder Layer |
| **Multi-Head Attention** | 8 heads, d_model = 512 |
| **Feed-Forward** | d_model = 512, d_ff = 2048 |
| **Positional Encoding** | Sinusoidal |
| **Embedding** | vocab_size × 512 |
| **Regularization** | Dropout 0.1, Label Smoothing 0.1 |
| **Optimizer** | Scheduled Adam (warmup 4000 steps) |
| **Beam Search** | beam_width = 5 |

### 3.2 Case B - GNMT (Google Neural Machine Translation)

| Thành phần | Chi tiết |
|------------|----------|
| **Encoder** | 2 lớp LSTM (uni-directional) |
| **Decoder** | 2 lớp LSTM + Attention |
| **Hidden size** | 128 |
| **Attention** | Luong Attention (scaled_luong) |
| **Embedding** | vocab_size × 128 |
| **Regularization** | Dropout 0.2 |
| **Optimizer** | SGD, learning rate 1.0 |
| **Decay** | Halving mỗi 1K steps sau 8K steps |

## 4. Kết quả huấn luyện

| Metric | Case A (Transformer) | Case B (GNMT) |
|--------|---------------------|---------------|
| **BLEU (test, tst2013)** | **25.88%** | **5.2%** |
| **BLEU (dev, tst2012)** | (không tách dev riêng) | 5.0% (best checkpoint: 5.34% @ step 18000) |
| **Perplexity** | - | 35.18 (test) / 33.69 (dev) |
| **Số bước train** | 20 epochs (~50K iterations) | 20 000 steps (~16 epochs) |
| **Thời gian train** | Không log tổng thời gian (xem ghi chú bên dưới) | ~2h08p (17:15:12 → 19:23:37, CPU) |
| **Môi trường** | Google Colab (GPU) | Local (CPU) |
| **Model** | Train từ đầu (from scratch, 20 epochs) | Train từ đầu (from scratch, 20 000 steps) |

> **Ghi chú số liệu:** các con số trên lấy trực tiếp từ output đã chạy trong `demo_transformer.ipynb`
> (cell tính `bleu(...)` cuối cùng) và từ `nmt_model/hparams` + `nmt_model/log_1785320109`

## 5. Phân tích kết quả

### 5.1 Về BLEU score

- **Transformer** đạt **25.88%** BLEU trên test (tst2013), phù hợp với benchmark của IWSLT 2015 En-Vi
  (Luong & Manning 2015 báo cáo ~23-26% BLEU cho hướng En→Vi ở quy mô dữ liệu này).
- **GNMT** chỉ đạt **5.2%** BLEU sau 20 000 steps — thấp hơn nhiều so với benchmark gốc của `tensorflow/nmt` (README báo cáo ~23-25 BLEU cho cấu hình 2-layer sau khi train đủ, dùng `standard_hparams/iwslt15.json`), tức là mô hình **chưa hội tụ / thiếu train**, không phải do kiến trúc LSTM kém hơn về bản chất.

**Nguyên nhân chênh lệch:**
1. **Quy mô train chưa tương xứng với kiến trúc**: GNMT ở đây dùng cấu hình rất nhỏ (`num_units=128`, `optimizer=sgd`, `learning_rate=1.0`, không dùng `standard_hparams/iwslt15.json` của repo gốc vốn khuyến nghị `num_units=512` + nhiều tinh chỉnh khác) và dừng ở 20 000 steps (~16 epoch). BLEU vẫn đang tăng dần đến bước cuối (4.7 → 5.34 → 5.2 dao động quanh 5%), tức là **chưa hội tụ**, cần train thêm nhiều steps nữa mới phản ánh đúng khả năng thật của kiến trúc.
2. **Cả hai đều train from scratch**, không có mô hình nào dùng checkpoint dựng sẵn — nên chênh lệch BLEU phản ánh sự khác biệt về kiến trúc + hparams + số bước train.
3. **Kiến trúc**: về mặt lý thuyết, Transformer có khả năng học biểu diễn song song và biểu diễn ngữ cảnh dài tốt hơn LSTM, nhưng với kết quả thực nghiệm này, phần lớn khoảng cách BLEU đến từ (1) và (3) ở trên chứ chưa thể kết luận chắc chắn là do kiến trúc, vì GNMT chưa được train/tune đủ để so sánh công bằng.

### 5.2 Về Perplexity

- GNMT đạt perplexity ~35 (test) / ~34 (dev) ở bước cuối — giảm đều so với ~17 000 lúc khởi tạo, cho
  thấy mô hình đang học tốt nhưng vẫn còn xa mức perplexity thấp của các mô hình đã hội tụ đầy đủ.
- Perplexity giảm dần trong khi BLEU tăng chậm và dao động (4.7 -> 5.34 -> 5.2 giữa các checkpoint 18K-20K)
  củng cố nhận định ở trên: mô hình cần thêm steps để hội tụ, chưa nên dừng ở 20 000.

### 5.3 Ví dụ dịch

**Câu nguồn (En, dòng 6 - tst2013.en):** "My family was not poor, and myself, I had never experienced hunger."
**Bản dịch tham chiếu (Vi):** "Gia đình của tôi không nghèo, và bản thân tôi thì chưa từng phải chịu đói."

| Model | Câu dịch (Vi) |
|-------|---------------|
| Transformer (Case A) | "gia đình tôi không nghèo, và bản thân tôi, tôi chưa bao giờ trải qua nạn đói." |
| GNMT (Case B) | "Cha tôi không thể sống sót, nhưng tôi không có sự nghèo đói." |

Transformer cho câu dịch mạch lạc, sát nghĩa câu gốc — phù hợp với BLEU cao đo được. GNMT dịch sai lệch hẳn ý nghĩa câu gốc (chủ ngữ, hành động đều khác) dù ngữ pháp câu tiếng Việt vẫn tạm ổn — minh hoạ trực quan cho việc mô hình chưa hội tụ dù perplexity/BLEU đo được không quá tệ trên toàn tập.

## 6. Ưu và nhược điểm

### 6.1 Transformer

**Ưu điểm:**
- BLEU cao hơn đáng kể (25.88% vs 5.2% trong thực nghiệm này)
- Xử lý song song, train nhanh trên GPU
- Phù hợp với câu dài nhờ Self-Attention
- Khả năng học biểu diễn ngôn ngữ mạnh mẽ, hội tụ nhanh hơn trong cùng số epoch

**Nhược điểm:**
- Cần nhiều tài nguyên tính toán (GPU, RAM)
- Kiến trúc phức tạp, nhiều tham số (~60-70M)
- Khó debug và tinh chỉnh
- Nhạy với thiết lập tiền xử lý

### 6.2 GNMT (LSTM + Attention)

**Ưu điểm:**
- Kiến trúc đơn giản, dễ hiểu và triển khai
- Ít tham số hơn (~2-3M)
- Phù hợp với bài toán nhỏ, data ít
- Dễ dàng fine-tune và điều chỉnh

**Nhược điểm:**
- BLEU thấp nếu không train đủ lâu
- Xử lý tuần tự, chậm trên GPU
- Khó khăn khi tăng kích thước mô hình (gradient vanishing/exploding)
- Hiệu năng kém trên câu dài
