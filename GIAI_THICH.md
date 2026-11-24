# Giải thích dự án Gomoku Policy CNN

Tài liệu này giải thích từ đầu đến cuối những gì đã làm trong dự án: dữ liệu lấy từ đâu và được biến đổi ra sao, mô hình là gì, vì sao nó học được, toán học đằng sau, và các "tham số" nghĩa là gì.

> Mẹo đọc: mở file này bằng VS Code và nhấn `Ctrl+Shift+V` (Markdown Preview) để công thức toán hiển thị đẹp.

---

## Mục lục

1. [Tóm tắt những gì đã làm](#1-tóm-tắt-những-gì-đã-làm)
2. [Ý tưởng cốt lõi: học bắt chước](#2-ý-tưởng-cốt-lõi-học-bắt-chước)
3. [Luồng dữ liệu từ đầu đến cuối](#3-luồng-dữ-liệu-từ-đầu-đến-cuối)
4. ["Tham số" là gì?](#4-tham-số-là-gì)
5. [Mô hình: mạng nơ-ron tích chập (CNN)](#5-mô-hình-mạng-nơ-ron-tích-chập-cnn)
6. [Đầu ra và hàm mất mát: softmax + cross-entropy](#6-đầu-ra-và-hàm-mất-mát-softmax--cross-entropy)
7. [Mô hình học như thế nào: gradient descent và backpropagation](#7-mô-hình-học-như-thế-nào-gradient-descent-và-backpropagation)
8. [Các kỹ thuật phụ trợ](#8-các-kỹ-thuật-phụ-trợ)
9. [Đánh giá: train/val, top-1, top-5](#9-đánh-giá-trainval-top-1-top-5)
10. [Cái gì "tối ưu", cái gì không?](#10-cái-gì-tối-ưu-cái-gì-không)
11. [Bảng siêu tham số](#11-bảng-siêu-tham-số)
12. [Hướng phát triển tiếp](#12-hướng-phát-triển-tiếp)
13. [Từ điển thuật ngữ](#13-từ-điển-thuật-ngữ)

---

## 1. Tóm tắt những gì đã làm

| Bước | Việc đã làm | Kết quả |
|---|---|---|
| 1 | Tải file `gomocup2019results.zip` từ gomocup.org (36 MB) | `data/raw/gomocup2019results.zip` |
| 2 | Mở zip và phân tích định dạng file ván cờ `.psq` | Phát hiện: **Freestyle chơi trên bàn 20×20**, không phải 15×15 |
| 3 | Thống kê: bao nhiêu ván "vừa" khung 15×15, khai cuộc dài bao nhiêu nước | ~89% số ván vừa khung 15×15 |
| 4 | Viết `create_dataset.py`: chuyển ván cờ thành cặp (bàn cờ, nước đi) | `data/gomoku_15x15.npz`: 321.774 mẫu từ 9.246 ván |
| 5 | Kiểm tra tính đúng của dataset | Không nhãn nào rơi vào ô đã có quân; số quân hai bên luôn hợp lệ |
| 6 | Viết notebook `train_policy.ipynb` để train trên GPU Kaggle | Mô hình ResNet ~1,34 triệu tham số |
| 7 | Chạy thử notebook trên CPU với tập nhỏ (300 ván, 1 epoch) | Chạy hết từ đầu đến cuối, không lỗi |

Sau đó đã train thật trên GPU T4 của Kaggle (20 epoch, khoảng 23 phút): top-1 trên val đạt khoảng 54,7%, top-5 khoảng 86%, val loss khoảng 1,50.

---

## 2. Ý tưởng cốt lõi: học bắt chước

### 2.1 Bài toán

Ta có hàng nghìn ván cờ do các engine mạnh nhất thế giới (Embryo, Rapfi, Yixin, ...) đánh với nhau. Ý tưởng:

> **Cho mô hình xem một thế cờ, bắt nó đoán nước đi mà engine đã chọn.**

Nếu đoán đủ giống engine, mô hình tự nó sẽ đánh khá hay. Cách này gọi là **học có giám sát** (supervised learning), cụ thể là **học bắt chước** (imitation learning). Mô hình loại này gọi là **policy network** (mạng chính sách: "trong thế cờ này nên đi đâu").

AlphaGo (2016) cũng bắt đầu đúng như vậy: một policy network học từ 30 triệu thế cờ của người chơi giỏi, đạt 57% đoán đúng nước đi.

### 2.2 Vì sao là bài toán phân loại 225 lớp?

Bàn 15×15 có 225 ô. Nước đi tiếp theo là **một trong 225 ô**, nên ta coi mỗi ô là một "lớp", giống như phân loại ảnh chó/mèo nhưng có 225 loại thay vì 2.

Đánh số ô: ô ở hàng `row`, cột `col` có số thứ tự

$$\text{index} = \text{row} \times 15 + \text{col} \qquad (0 \le \text{index} \le 224)$$

Ví dụ ô giữa bàn (hàng 7, cột 7) là $7 \times 15 + 7 = 112$.

### 2.3 Vì sao hợp với người mới?

- **Dễ debug:** mỗi mẫu có đáp án rõ ràng; có thể vẽ bàn cờ ra xem mô hình đoán gì, đúng hay sai.
- **Train nhanh:** chỉ cần vài chục phút GPU, khác với tự đánh với chính mình (reinforcement learning) cần hàng nghìn giờ.
- **Khung chuẩn:** dữ liệu → mô hình → loss → tối ưu, đúng khung của hầu hết bài toán deep learning.

---

## 3. Luồng dữ liệu từ đầu đến cuối

```
gomocup2019results.zip  (40.417 file, gồm 10.224 ván Freestyle)
        │
        │  create_dataset.py   (chạy trên máy bạn)
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Đọc file .psq → danh sách nước đi (x, y, thời gian)      │
│ 2. Ván không vừa 15×15 → bỏ; ván vừa → dời vào giữa bàn     │
│ 3. Phát lại ván từng nước, trước mỗi nước chụp "ảnh" bàn cờ │
│ 4. Lưu (ảnh bàn cờ, nước đi, id ván) vào file .npz          │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
gomoku_15x15.npz   X: (321774, 2, 15, 15)   y: (321774,)   game: (321774,)
        │
        │  upload lên Kaggle → train_policy.ipynb   (chạy trên GPU)
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Chia train/val theo ván, bỏ thế cờ val trùng với train   │
│ 6. Đưa hết dữ liệu lên GPU                                  │
│ 7. Mỗi bước: lấy 512 mẫu → xoay/lật ngẫu nhiên              │
│ 8. Mô hình → 225 điểm số → che ô đã có quân → softmax       │
│ 9. So với nhãn → loss → backprop → cập nhật tham số         │
│ 10. Hết mỗi epoch: đo trên val, lưu bản tốt nhất            │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
gomoku_policy.pt   (bộ tham số đã học)  →  best_move(board, player)
```

### 3.1 Định dạng file ván cờ `.psq`

Một file thật trong dataset (`Freestyle1/0_0_10_2.psq`), rút gọn:

```
Piskvorky 20x20, 11:11, 0      ← dòng tiêu đề: bàn 20×20
6,11,0                         ← nước 1: x=6, y=11, thời gian nghĩ 0 ms
6,10,0                         ← nước 2   (các nước có thời gian 0 = khai cuộc do giải đặt sẵn)
...                            (10 nước khai cuộc)
8,10,22817                     ← nước 11: engine nghĩ 22,8 giây
5,11,15459
...
YIXIN18.zip                    ← engine đi trước
WHOSE19.zip                    ← engine đi sau
```

Toạ độ đếm từ 1; script trừ 1 để đếm từ 0 như trong Python.

### 3.2 Từ bàn 20×20 sang 15×15

Với mỗi ván, script tìm **khung chữ nhật nhỏ nhất chứa mọi quân** (bounding box):

- Nếu khung rộng hoặc cao hơn 15 → ván này không đặt vừa bàn 15×15, **bỏ** (978 ván, ~10%).
- Nếu vừa → **dịch** cả ván sao cho khung nằm giữa bàn 15×15.

Phép dịch giữ nguyên hình dạng các thế cờ, mà luật Gomoku chỉ quan tâm hình dạng (5 quân liền nhau), nên ván cờ vẫn hợp lệ.

Một điểm cần biết: engine đánh ván đó trên bàn 20×20, nên nó không "biết" mép bàn 15×15. Với những ván đã lọc thì chuyện này hiếm khi ảnh hưởng, vì toàn bộ ván diễn ra trong khung 15×15.

### 3.3 Phát lại ván và tạo mẫu

Script đặt lần lượt từng quân lên bàn. **Trước** mỗi nước (trừ nước khai cuộc), nó chụp lại bàn cờ:

```
bàn cờ trước nước thứ i   →   X[i]    (đầu vào)
nước thứ i                →   y[i]    (nhãn, 0..224)
số thứ tự ván             →   game[i]
```

**Vì sao bỏ nước khai cuộc khỏi nhãn?** Các nước đó do ban tổ chức đặt, không phải engine chọn. Học chúng chẳng dạy được gì về cách đánh.

**Mã hoá bàn cờ "nhìn từ phía người sắp đi":** mỗi bàn cờ là 2 tấm 15×15 chỉ gồm số 0/1:

- Tấm 0: quân **của người sắp đi** (1 = có quân).
- Tấm 1: quân **của đối thủ**.

Ví dụ (X = tấm 0, O = tấm 1, `*` = nước engine đã đi):

```
......X..O.....
......OOX......
......XOO*.....
.....XO.X......
....XOXX.O.....
```

Nhờ cách mã hoá này, mô hình không cần quan tâm mình là đen hay trắng: lúc nào nó cũng trả lời câu hỏi "**tôi** nên đi đâu?". Số mẫu cho đen và trắng được gộp chung, nên dữ liệu được dùng hiệu quả gấp đôi.

### 3.4 Kích thước dữ liệu

| Mảng | Shape | Kiểu | Ý nghĩa |
|---|---|---|---|
| `X` | (321774, 2, 15, 15) | uint8 | 321.774 bàn cờ, mỗi bàn 2 tấm 15×15 |
| `y` | (321774,) | int16 | nước đi đúng, 0..224 |
| `game` | (321774,) | int32 | mẫu thuộc ván nào |

Chưa nén: 321.774 × 450 byte ≈ 145 MB. Nén lại chỉ còn 2,6 MB, vì bàn cờ chủ yếu là số 0.

Tuỳ chọn `--winner-only` chỉ giữ nước đi của người thắng, được 161.604 mẫu. Ít dữ liệu hơn nhưng "chất lượng" hơn: nước của người thua đôi khi chính là nước dẫn đến thua.

---

## 4. "Tham số" là gì?

Đây là khái niệm quan trọng nhất, nên giải thích kỹ.

### 4.1 Ví dụ đơn giản nhất

Giả sử ta muốn dự đoán giá nhà $y$ từ diện tích $x$ bằng một đường thẳng:

$$y = a \cdot x + b$$

- $x$ là **đầu vào** (cho trước).
- $y$ là **đầu ra** (cần đoán).
- $a$ và $b$ là **tham số** (parameters): những con số mà **máy tự tìm** sao cho đường thẳng khớp dữ liệu nhất.

"Huấn luyện" (training) nghĩa là: bắt đầu với $a, b$ ngẫu nhiên, rồi chỉnh dần chúng để sai số giảm. Sau khi train xong, $a, b$ chứa "kiến thức" mô hình học được.

### 4.2 Mạng nơ-ron = rất nhiều tham số

Mạng nơ-ron cũng là một hàm $f(x; \theta)$, chỉ là phức tạp hơn nhiều, với $\theta$ là **hàng triệu tham số** thay vì 2. Mô hình của dự án có **1.340.737 tham số**. Mỗi tham số là một số thực, và **toàn bộ "trí khôn" của mô hình nằm trong 1,34 triệu con số đó**. File `gomoku_policy.pt` chính là nơi lưu chúng.

Đếm cụ thể (giải thích các lớp ở mục 5):

| Phần | Cách tính | Số tham số |
|---|---|---|
| Stem conv 5×5, 3 → 96 kênh | $3 \times 96 \times 5 \times 5$ | 7.200 |
| Stem BatchNorm | $2 \times 96$ ($\gamma, \beta$) | 192 |
| 1 conv 3×3, 96 → 96 | $96 \times 96 \times 3 \times 3$ | 82.944 |
| 1 ResBlock = 2 conv + 2 BN | $2 \times 82.944 + 2 \times 192$ | 166.272 |
| 8 ResBlock | $8 \times 166.272$ | 1.330.176 |
| Head conv 1×1, 96 → 32 + BN | $96 \times 32 + 64$ | 3.136 |
| Head conv 1×1, 32 → 1 (có bias) | $32 + 1$ | 33 |
| **Tổng** | | **1.340.737** |

Hơn 99% tham số nằm trong 8 ResBlock: đó là "bộ não" chính.

### 4.3 Tham số và siêu tham số (hyperparameters)

Hai khái niệm này rất hay bị nhầm:

| | **Tham số** (parameters) | **Siêu tham số** (hyperparameters) |
|---|---|---|
| Ai đặt? | Máy tự học qua training | **Con người** chọn trước khi train |
| Bao nhiêu? | 1,34 triệu | Khoảng 10 con số |
| Ví dụ | Các trọng số trong bộ lọc conv | Số epoch, learning rate, số kênh, số block |
| Thay đổi khi nào? | Liên tục trong lúc train | Chỉ khi bạn sửa code |

Với ví dụ đường thẳng: $a, b$ là tham số; còn "chỉnh $a, b$ bao nhiêu lần" hay "mỗi lần chỉnh mạnh cỡ nào" là siêu tham số.

Bảng đầy đủ các siêu tham số của dự án nằm ở [mục 11](#11-bảng-siêu-tham-số).

---

## 5. Mô hình: mạng nơ-ron tích chập (CNN)

### 5.1 Phép tích chập (convolution) là gì?

Một **bộ lọc** (kernel) 3×3 là một ô vuông 9 con số (9 tham số). Nó **trượt** qua từng vị trí trên bàn cờ. Tại mỗi vị trí, nó nhân 9 con số của mình với 9 ô bên dưới rồi cộng lại.

Ví dụ một bộ lọc "phát hiện 3 quân thẳng hàng ngang":

```
Vùng bàn cờ      Bộ lọc (tham số)       Nhân từng ô rồi cộng
0 0 0            0 0 0
1 1 1     ×      1 1 1        =   1+1+1 = 3   ← giá trị cao: "có mẫu này ở đây!"
0 0 0            0 0 0
```

Tại vùng không có 3 quân ngang, kết quả thấp. Kết quả trượt khắp bàn tạo ra một **bản đồ đặc trưng** (feature map) 15×15 cho biết "mẫu này xuất hiện ở đâu".

Công thức: với đầu vào $x$ có $C_{in}$ kênh và bộ lọc $w$,

$$\text{out}[r, c] = \sum_{k=1}^{C_{in}} \sum_{i=-1}^{1} \sum_{j=-1}^{1} w[k, i, j] \cdot x[k, r+i, c+j]$$

Một lớp conv 96 → 96 kênh có **96 bộ lọc**, mỗi bộ lọc nhìn cả 96 kênh đầu vào. Vì vậy mới có $96 \times 96 \times 9$ tham số.

**Điều quan trọng: các bộ lọc không do ai thiết kế.** Ban đầu chúng là số ngẫu nhiên. Qua training, máy tự tìm ra những bộ lọc hữu ích, ví dụ phát hiện "ba mở", "bốn", "hai đường giao nhau".

### 5.2 Vì sao CNN hợp với cờ?

1. **Tính cục bộ:** các thế cờ quan trọng (ba, bốn, chặn) nằm trong vùng nhỏ. Bộ lọc 3×3 nhìn đúng vùng nhỏ đó.
2. **Bất biến khi dịch chuyển:** "bốn mở" ở góc trái hay giữa bàn đều nguy hiểm như nhau. Cùng một bộ lọc được dùng ở mọi vị trí (**chia sẻ trọng số**), nên học một lần là dùng được khắp bàn.
3. **Tiết kiệm tham số:** một lớp "nối đầy đủ" (fully-connected) từ 225 ô sang 225 ô cần 50.625 tham số cho mỗi kênh. Conv 3×3 chỉ cần 9.

### 5.3 Xếp chồng lớp: từ mẫu nhỏ đến chiến lược

- Lớp 1 nhìn vùng 5×5 (stem dùng bộ lọc 5×5).
- Mỗi lớp conv 3×3 tiếp theo nhìn rộng thêm 1 ô mỗi phía.
- Mô hình có 1 conv 5×5 + 16 conv 3×3, nên **vùng nhìn** (receptive field) có bán kính $2 + 16 = 18$ ô, tức 37×37: **phủ toàn bộ bàn 15×15**.

Lớp nông phát hiện mẫu đơn giản ("hai quân liền"), lớp sâu kết hợp chúng thành khái niệm phức tạp ("nếu đi đây thì tạo được hai đường ba cùng lúc").

### 5.4 Kiến trúc cụ thể

```
Đầu vào (B, 2, 15, 15)          B = số bàn cờ trong 1 batch
   │ + thêm kênh toàn số 1
   ▼
(B, 3, 15, 15)
   │ Stem: Conv 5×5 → BatchNorm → ReLU
   ▼
(B, 96, 15, 15)                 96 bản đồ đặc trưng
   │ ResBlock × 8
   ▼
(B, 96, 15, 15)
   │ Head: Conv 1×1 (96→32) → BN → ReLU → Conv 1×1 (32→1)
   ▼
(B, 1, 15, 15) → làm phẳng → (B, 225)    225 "điểm số" (logit), mỗi ô một điểm
   │ ô đã có quân → điểm = −10⁹
   ▼
softmax → 225 xác suất
```

Kích thước 15×15 được **giữ nguyên suốt mạng** (padding giữ kích thước, không dùng pooling), vì đầu ra cần đúng 15×15 ô. Đây là khác biệt với mạng phân loại ảnh thông thường.

**Kênh toàn số 1:** khi bộ lọc trượt ra ngoài mép bàn, phần ngoài được điền số 0 (zero-padding). Không có kênh này thì "ô trống" và "ngoài bàn" đều là 0, mô hình không phân biệt được. Có kênh số 1, ngoài bàn là 0 còn trong bàn là 1, nên mô hình "thấy" mép bàn.

**Che ô đã có quân (masking):** gán điểm −10⁹ cho ô có quân. Sau softmax, xác suất của ô đó ≈ 0. Mô hình không bao giờ đi vào ô đã có quân và không phải tốn công học luật này.

### 5.5 ReLU: nguồn gốc của "phi tuyến"

$$\text{ReLU}(z) = \max(0, z)$$

Nếu chỉ xếp chồng các phép nhân-cộng (tuyến tính) thì dù 100 lớp cũng chỉ tương đương **một** phép tuyến tính, vì hợp của các hàm tuyến tính vẫn là hàm tuyến tính. ReLU "bẻ cong" hàm, cho phép mạng biểu diễn những quy tắc kiểu "**nếu** có ba mở **và** đối thủ không có bốn **thì**...". Định lý xấp xỉ phổ quát (universal approximation) nói rằng mạng đủ lớn có ReLU có thể xấp xỉ bất kỳ hàm liên tục nào.

### 5.6 BatchNorm: giữ số liệu ở thang đo ổn định

Với mỗi kênh, BatchNorm chuẩn hoá giá trị trong batch về trung bình 0, phương sai 1, rồi co giãn lại bằng 2 tham số học được $\gamma, \beta$:

$$\hat{z} = \frac{z - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}, \qquad \text{out} = \gamma \hat{z} + \beta$$

Tác dụng: số liệu không phình to hay teo nhỏ qua nhiều lớp, nên có thể dùng learning rate lớn hơn và train nhanh, ổn định hơn. Khi dự đoán (`model.eval()`), BN dùng trung bình/phương sai tích luỹ trong lúc train thay vì của batch.

### 5.7 Residual connection: vì sao mạng sâu vẫn train được

Mỗi ResBlock tính:

$$\text{out} = \text{ReLU}\big(x + F(x)\big)$$

trong đó $F$ là 2 lớp conv. Thay vì học thẳng đầu ra, block chỉ học **phần chỉnh sửa** $F(x)$ cộng thêm vào $x$.

Vì sao điều này quan trọng? Khi lan truyền ngược (mục 7), đạo hàm qua block có dạng

$$\frac{\partial\, \text{out}}{\partial x} \approx 1 + \frac{\partial F}{\partial x}$$

Số **1** là "đường cao tốc" để tín hiệu học truyền thẳng qua nhiều lớp mà không bị nhân với các số nhỏ liên tiếp đến mức biến mất (**vanishing gradient**). Nhờ vậy ResNet (2015) mới train được mạng hàng trăm lớp. AlphaGo Zero dùng đúng loại block này.

---

## 6. Đầu ra và hàm mất mát: softmax + cross-entropy

### 6.1 Softmax: từ điểm số sang xác suất

Mạng cho ra 225 số thực bất kỳ $z_1, \ldots, z_{225}$ (gọi là **logit**). Softmax biến chúng thành xác suất:

$$p_k = \frac{e^{z_k}}{\sum_{j=1}^{225} e^{z_j}}$$

Mỗi $p_k > 0$ và tổng các $p_k$ bằng 1. Điểm cao hơn thì xác suất cao hơn, và hàm mũ $e^z$ khuếch đại khoảng cách giữa các điểm.

Ví dụ với 3 ô có điểm $[2, 1, 0]$:

| | ô A | ô B | ô C |
|---|---|---|---|
| logit $z$ | 2 | 1 | 0 |
| $e^z$ | 7,39 | 2,72 | 1,00 |
| $p = e^z / 11{,}11$ | **0,665** | 0,245 | 0,090 |

Ô bị che có $z = -10^9$, nên $e^z = 0$ và xác suất bằng 0.

### 6.2 Cross-entropy: đo "sai" bao nhiêu

Với nhãn đúng là ô $y$:

$$\mathcal{L} = -\log p_y$$

- Mô hình chắc chắn đúng ($p_y = 1$): $\mathcal{L} = 0$.
- Ví dụ trên, nếu đáp án là A: $-\ln 0{,}665 = 0{,}41$. Nếu đáp án là C: $-\ln 0{,}09 = 2{,}41$, phạt nặng vì mô hình cho C rất ít khả năng.
- Mô hình đoán mò đều 225 ô: $\mathcal{L} = \ln 225 \approx 5{,}42$. **Đây là mức loss lúc bắt đầu train**, và mọi con số thấp hơn 5,42 là mô hình đã học được gì đó.

Loss của một batch là trung bình loss của 512 mẫu.

### 6.3 Vì sao dùng cross-entropy? Nguyên lý toán học

**(a) Đó là ước lượng hợp lý cực đại (Maximum Likelihood).** Coi mô hình là một phân phối xác suất $p_\theta(\text{nước đi} \mid \text{bàn cờ})$. Xác suất mô hình "sinh ra" đúng toàn bộ dataset là

$$\prod_{i=1}^{M} p_\theta(y_i \mid x_i)$$

Tìm $\theta$ làm tích này lớn nhất (giải thích dữ liệu tốt nhất) tương đương với lấy $\log$ rồi đổi dấu, tức là **làm nhỏ nhất tổng cross-entropy**:

$$\arg\max_\theta \prod_i p_\theta(y_i \mid x_i) = \arg\min_\theta \sum_i -\log p_\theta(y_i \mid x_i)$$

Ước lượng hợp lý cực đại có tính chất **nhất quán** (consistent): khi dữ liệu tiến tới vô hạn và mô hình đủ mạnh, nó hội tụ về phân phối thật.

**(b) Cực tiểu đạt được khi mô hình khớp đúng phân phối thật.** Gọi $q(\cdot \mid x)$ là phân phối thật của "engine đi đâu ở thế cờ $x$". Kỳ vọng của loss là

$$\mathbb{E}[\mathcal{L}] = \underbrace{H(q)}_{\text{không đổi}} + \underbrace{D_{KL}(q \,\|\, p_\theta)}_{\ge 0}$$

Phân kỳ KL $D_{KL} \ge 0$ và chỉ bằng 0 khi $p_\theta = q$. Vậy loss nhỏ nhất **đúng khi** mô hình dự đoán đúng phân phối của engine. Hàm loss có tính chất này gọi là **proper scoring rule**: nó không khuyến khích mô hình "nói dối" về độ tự tin.

Ví dụ: nếu ở một thế cờ, engine đi ô A 70% số lần và ô B 30%, loss nhỏ nhất đạt được khi mô hình nói đúng "A: 0,7; B: 0,3", chứ không phải "A: 1,0".

**(c) Đạo hàm cực kỳ gọn.** Đạo hàm của loss theo logit:

$$\frac{\partial \mathcal{L}}{\partial z_k} = p_k - \mathbb{1}[k = y]$$

Tức là "xác suất dự đoán trừ đáp án". Ô đúng bị đẩy điểm lên (đạo hàm $p_y - 1 < 0$), các ô sai bị kéo điểm xuống tỉ lệ với mức mô hình đã tin nhầm vào chúng. Đạo hàm không bao giờ "bão hoà" khi mô hình sai nặng, khác với nếu dùng sai số bình phương, nên học nhanh.

---

## 7. Mô hình học như thế nào: gradient descent và backpropagation

### 7.1 Gradient descent: đi xuống dốc

Hình dung loss là độ cao của một vùng đồi núi, mỗi điểm trên đó tương ứng với một bộ 1,34 triệu tham số. Ta muốn xuống chỗ thấp nhất.

**Gradient** $\nabla_\theta \mathcal{L}$ là vector 1,34 triệu chiều chỉ hướng dốc **lên** nhanh nhất. Đi ngược hướng đó:

$$\theta \leftarrow \theta - \eta \, \nabla_\theta \mathcal{L}$$

$\eta$ là **learning rate** (tốc độ học), một siêu tham số quyết định mỗi bước đi dài bao nhiêu:
- Quá lớn: nhảy qua đáy, loss dao động hoặc tăng vọt.
- Quá nhỏ: đi rất chậm.

### 7.2 Backpropagation: tính gradient thế nào?

Mạng là chuỗi hàm lồng nhau: $\mathcal{L} = \text{CE}(\text{head}(\text{block}_8(\cdots\text{stem}(x))))$. Theo **quy tắc dây chuyền** (chain rule) của đạo hàm:

$$\frac{\partial \mathcal{L}}{\partial \theta_{\text{stem}}} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial h_8} \cdots \frac{\partial h_1}{\partial \theta_{\text{stem}}}$$

Backpropagation tính các đạo hàm này **từ lớp cuối ngược về lớp đầu**, tái sử dụng kết quả trung gian, nên tính gradient cho cả 1,34 triệu tham số chỉ tốn cỡ gấp 2 lần một lượt tính xuôi. PyTorch làm việc này tự động khi gọi `loss.backward()`.

### 7.3 Mini-batch: không cần nhìn hết dữ liệu mới đi một bước

Tính gradient trên cả 290 nghìn mẫu cho mỗi bước thì quá chậm. Thay vào đó, mỗi bước lấy ngẫu nhiên **512 mẫu** (một batch) và tính gradient trên đó:

- Gradient batch là **ước lượng không chệch** của gradient thật: trung bình thì đúng hướng.
- Độ nhiễu giảm theo $1/\sqrt{B}$. 512 là đủ ổn định và vẫn tận dụng tốt GPU.
- Chút nhiễu còn **có lợi**: giúp thoát khỏi những "hố" hẹp, thường tìm được nghiệm tổng quát hoá tốt hơn.

**Epoch** = một lượt đi qua toàn bộ dữ liệu train, khoảng 566 bước với batch 512. Notebook train 20 epoch, tức khoảng 11.300 bước cập nhật.

### 7.4 Adam / AdamW: gradient descent thông minh hơn

Gradient descent thuần dùng chung một learning rate cho mọi tham số. **Adam** cải tiến bằng hai ý:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(quán tính: trung bình trượt của gradient)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(độ lớn: trung bình trượt của gradient bình phương)}$$
$$\theta \leftarrow \theta - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

($\hat{m}, \hat{v}$ là $m, v$ đã hiệu chỉnh vì khởi đầu từ 0.)

- **Quán tính $m$:** như hòn bi lăn, giữ đà theo hướng nhất quán và triệt tiêu dao động qua lại.
- **Chia cho $\sqrt{v}$:** tham số nào có gradient lớn thì bước nhỏ lại, gradient nhỏ thì bước to ra. Mỗi tham số có "learning rate riêng" tự điều chỉnh.

**AdamW** thêm **weight decay**: mỗi bước kéo nhẹ mọi tham số về 0,

$$\theta \leftarrow \theta - \eta \lambda \theta, \qquad \lambda = 10^{-4}$$

Đây là một dạng **regularization** (điều chuẩn): ngăn tham số phình to để "học thuộc lòng" dữ liệu train, nhờ đó mô hình đoán tốt hơn trên thế cờ chưa gặp.

### 7.5 OneCycle: thay đổi learning rate theo thời gian

Learning rate không cố định mà thay đổi theo lịch:

```
lr
2e-3 ┤     ╭──╮
     │    ╱    ╲___
     │   ╱         ╲___
     │  ╱              ╲___
8e-5 ┤─╯                   ╲____ (gần 0)
     └──┬───────────────────────── bước
       10%                    100%
     khởi động     giảm dần theo đường cosin
```

- **Khởi động (10% đầu):** tham số ban đầu ngẫu nhiên, gradient lộn xộn, nên đi bước nhỏ cho an toàn rồi tăng dần.
- **Giữa:** learning rate cao để khám phá nhanh vùng tốt.
- **Cuối:** giảm dần về gần 0 để "đáp" chính xác vào đáy, giống đi chậm lại khi đến gần đích.

OneCycle trong PyTorch đồng thời điều chỉnh ngược chiều hệ số quán tính $\beta_1$ của Adam (0,85 ↔ 0,95).

---

## 8. Các kỹ thuật phụ trợ

### 8.1 Tăng cường dữ liệu bằng 8 phép đối xứng

Luật Gomoku không đổi khi **xoay** bàn cờ (0°, 90°, 180°, 270°) hay **lật** gương. Có tổng cộng $4 \times 2 = 8$ phép biến đổi (trong toán học gọi là nhóm nhị diện $D_4$). Nếu nước X là tốt ở một thế cờ, thì ở thế cờ xoay 90°, nước X xoay 90° cũng tốt y như vậy.

Notebook áp dụng: mỗi batch chọn ngẫu nhiên 1 trong 8 phép, biến đổi **cả bàn cờ lẫn nhãn**. Nhãn được ánh xạ lại bằng bảng tra `inv[k]`, tính sẵn bằng cách áp phép biến đổi lên lưới số thứ tự 0..224.

Lợi ích:
- Dữ liệu hiệu dụng **gấp 8 lần** (~2,6 triệu mẫu) mà không tốn thêm bộ nhớ.
- Mô hình học được tính đối xứng thay vì "thuộc" một hướng cụ thể.

Notebook có đoạn kiểm tra tự động (`symmetry OK`) để chắc chắn nhãn biến đổi khớp với bàn cờ, vì đây là chỗ rất dễ sai mà không ai nhận ra.

### 8.2 Mixed precision (fp16)

Mặc định máy tính dùng số thực 32-bit (fp32). GPU T4 tính số 16-bit (fp16) **nhanh hơn nhiều lần** và tốn một nửa bộ nhớ. `torch.autocast` tự động dùng fp16 cho các phép conv và giữ fp32 ở những chỗ cần chính xác.

Vấn đề: fp16 không biểu diễn được số quá nhỏ (dưới khoảng $6 \times 10^{-8}$). Gradient nhỏ sẽ bị làm tròn thành 0. **GradScaler** giải quyết bằng cách nhân loss với một số lớn $S$ trước khi tính gradient (gradient cũng được nhân $S$, nên không bị làm tròn về 0), rồi chia lại cho $S$ trước khi cập nhật tham số. Nếu bị tràn số (inf), bước đó được bỏ qua và $S$ giảm đi.

### 8.3 Để toàn bộ dữ liệu trên GPU

Dữ liệu chỉ khoảng 145 MB, nhỏ so với 16 GB của T4. Đưa hết lên GPU một lần thì mỗi bước chỉ cần lấy chỉ mục, không phải chép dữ liệu CPU → GPU liên tục. Đơn giản hơn và nhanh hơn dùng `DataLoader`.

---

## 9. Đánh giá: train/val, top-1, top-5

### 9.1 Vì sao cần tập validation?

Mục tiêu là mô hình đánh tốt ở **thế cờ mới**, không phải thuộc lòng thế cờ đã thấy. Vì vậy giữ riêng 10% số ván (**val**) không cho mô hình học, chỉ dùng để chấm điểm.

- **Train loss giảm, val loss cũng giảm:** đang học tốt.
- **Train loss giảm, val loss tăng:** **overfitting** (học vẹt). Notebook luôn lưu bản có val tốt nhất nên tránh được việc giữ lại bản đã học vẹt.

### 9.2 Hai cái bẫy "lộ đề" đã được xử lý

1. **Chia theo ván, không theo mẫu.** Nước 20 và nước 21 trong cùng một ván có bàn cờ gần như giống hệt nhau. Nếu chia ngẫu nhiên từng mẫu, val sẽ chứa những thế cờ "anh em" với train, và điểm val cao giả tạo.
2. **Bỏ thế cờ val trùng hẳn với train.** Nhiều ván dùng chung khai cuộc, nên ~11% thế cờ xuất hiện ở nhiều ván. Những thế cờ val đã có trong train bị loại khỏi phần chấm điểm.

### 9.3 Các chỉ số

- **Top-1 accuracy:** tỉ lệ ô có xác suất cao nhất trùng đúng nước engine đã đi.
- **Top-5 accuracy:** tỉ lệ nước của engine nằm trong 5 ô mô hình đánh giá cao nhất.
- **Val loss:** cross-entropy trung bình trên val. Mức xuất phát khi đoán mò là 5,42.

Kết quả thật sau 20 epoch: top-1 khoảng 54,7%, top-5 khoảng 86%, val loss khoảng 1,50, train loss 1,42 (không học vẹt). Đừng mong 100%: ở nhiều thế cờ có vài nước tốt ngang nhau, và các engine khác nhau chọn khác nhau.

---

## 10. Cái gì "tối ưu", cái gì không?

Cần nói rõ để tránh hiểu lầm.

**Tối ưu, có cơ sở toán học:**
- Cross-entropy là lựa chọn đúng cho bài toán: cực tiểu của nó là khi mô hình khớp đúng phân phối nước đi của engine (mục 6.3).
- Gradient luôn chỉ đúng hướng giảm loss tại điểm hiện tại; backprop tính nó chính xác và hiệu quả.

**Không đảm bảo tối ưu:**
- **Bài toán không lồi:** vùng đồi núi của loss có rất nhiều thung lũng. Gradient descent tìm được một đáy **tốt**, không đảm bảo đáy **thấp nhất**. Trong thực tế, với mạng lớn, các đáy tìm được thường tốt gần như nhau.
- **Siêu tham số** (96 kênh, 8 block, lr 2e-3, ...) là các giá trị mặc định hợp lý theo kinh nghiệm, **không phải** giá trị đã chứng minh là tốt nhất. Có thể thử nghiệm để cải thiện.
- **Bắt chước ≠ chơi tối ưu:** mô hình học **giống engine** (kể cả nước sai của engine), chứ không học **đánh thắng**. Trần năng lực của nó là các engine trong dữ liệu.
- **Không tính trước:** mô hình chỉ "nhìn một lần rồi đi" (giống trực giác), không thử "nếu tôi đi đây, đối thủ đi kia...". Vì vậy nó có thể bỏ sót đòn chiến thuật mà một chương trình tìm kiếm đơn giản thấy ngay.

---

## 11. Bảng siêu tham số

Những con số **bạn** có thể chỉnh trong notebook:

| Siêu tham số | Giá trị | Ý nghĩa | Tăng lên thì | Giảm xuống thì |
|---|---|---|---|---|
| `EPOCHS` | 20 | Số lượt đi qua toàn bộ dữ liệu | Học kỹ hơn, lâu hơn; quá nhiều có thể overfit | Nhanh hơn, có thể chưa học xong |
| `BS` (batch size) | 512 | Số mẫu mỗi bước | Gradient ổn định hơn, ít bước hơn mỗi epoch | Nhiễu hơn, nhiều bước hơn |
| `max_lr` | 2e-3 | Learning rate đỉnh | Học nhanh, dễ mất ổn định | Ổn định, chậm |
| `weight_decay` | 1e-4 | Mức kéo tham số về 0 | Chống overfit mạnh hơn, có thể underfit | Dễ overfit hơn |
| `pct_start` | 0.1 | Tỉ lệ bước dùng để khởi động lr | Khởi động êm hơn | Vào lr cao sớm hơn |
| `channels` | 96 | Số bộ lọc mỗi lớp ("độ rộng") | Nhận ra nhiều loại mẫu hơn; tham số tăng theo bình phương | Mô hình nhỏ, nhanh, yếu hơn |
| `blocks` | 8 | Số ResBlock ("độ sâu") | Suy luận nhiều tầng hơn; chậm hơn | Nhanh, nông hơn |
| Tỉ lệ val | 10% số ván | Phần dữ liệu để chấm điểm | Chấm điểm tin cậy hơn, ít dữ liệu train | Ngược lại |
| `DATA_NAME` | `gomoku_15x15.npz` | Dataset dùng để train | — | Đổi sang `..._winner.npz` để chỉ học người thắng |

Trong `create_dataset.py`:

| Tuỳ chọn | Mặc định | Ý nghĩa |
|---|---|---|
| `--leagues` | Freestyle1–4 | Giải nào được dùng |
| `--winner-only` | tắt | Chỉ lấy nước của người thắng |

**Gợi ý thử nghiệm** (mỗi lần đổi một thứ để biết thứ nào có tác dụng):
1. Chạy mặc định, ghi lại top-1 và top-5 của val.
2. Thử `winner_only`.
3. Thử mô hình lớn hơn: `channels=128, blocks=10`.
4. Nếu val loss bắt đầu tăng trong khi train loss vẫn giảm, giảm `EPOCHS` hoặc tăng `weight_decay`.

---

## 12. Hướng phát triển tiếp

Theo thứ tự từ dễ đến khó:

1. **Thêm kiểm tra chiến thuật bắt buộc:** trước khi gọi mô hình, nếu có nước thắng ngay thì đi; nếu đối thủ sắp thắng thì chặn. Rất dễ làm và vá được điểm yếu lớn nhất.
2. **Value head:** thêm một đầu ra thứ hai dự đoán "ai sẽ thắng từ thế cờ này" (dữ liệu đã có kết quả ván).
3. **Kết hợp tìm kiếm:** dùng policy để chọn vài nước triển vọng, rồi tìm kiếm sâu (alpha-beta hoặc MCTS) chỉ trên những nước đó. Đây là cách AlphaGo kết hợp "trực giác" với "tính toán".
4. **Tự đánh với chính mình** (reinforcement learning, kiểu AlphaZero): vượt qua trần năng lực của dữ liệu engine. Tốn nhiều GPU hơn hẳn.

---

## 13. Từ điển thuật ngữ

| Thuật ngữ | Nghĩa ngắn gọn |
|---|---|
| **Tham số (parameter)** | Con số bên trong mô hình, do máy tự học. Mô hình này có 1,34 triệu. |
| **Siêu tham số (hyperparameter)** | Con số người chọn trước khi train (epoch, lr, số kênh...). |
| **Mẫu (sample)** | Một cặp (bàn cờ, nước đi đúng). |
| **Nhãn (label)** | Đáp án đúng của một mẫu: số 0..224. |
| **Batch** | Nhóm mẫu xử lý cùng lúc trong một bước (512). |
| **Epoch** | Một lượt đi qua toàn bộ dữ liệu train. |
| **Logit** | Điểm số thô mô hình cho mỗi ô, trước softmax. |
| **Softmax** | Hàm biến điểm số thành xác suất có tổng bằng 1. |
| **Loss** | Con số đo mô hình sai bao nhiêu; train là làm nó nhỏ đi. |
| **Cross-entropy** | Loss $-\log p_{\text{đúng}}$ cho bài toán phân loại. |
| **Gradient** | Hướng làm loss tăng nhanh nhất; đi ngược nó để giảm loss. |
| **Backpropagation** | Thuật toán tính gradient từ lớp cuối ngược về lớp đầu. |
| **Learning rate** | Độ dài mỗi bước cập nhật tham số. |
| **Optimizer** | Quy tắc cập nhật tham số từ gradient (ở đây là AdamW). |
| **Conv / tích chập** | Phép trượt bộ lọc nhỏ qua bàn cờ để phát hiện mẫu. |
| **Feature map** | Bản đồ 15×15 cho biết một mẫu xuất hiện ở đâu. |
| **Kênh (channel)** | Một tấm 15×15 trong dữ liệu; mỗi bộ lọc tạo ra một kênh. |
| **ResBlock** | Khối 2 lớp conv có đường tắt $x + F(x)$. |
| **BatchNorm** | Lớp chuẩn hoá giá trị để train ổn định. |
| **ReLU** | Hàm $\max(0, z)$, tạo tính phi tuyến. |
| **Overfitting** | Mô hình thuộc lòng dữ liệu train, đoán kém dữ liệu mới. |
| **Regularization** | Kỹ thuật chống overfitting (ví dụ weight decay, tăng cường dữ liệu). |
| **Data augmentation** | Tạo thêm dữ liệu bằng biến đổi hợp lệ (xoay, lật). |
| **Validation (val)** | Dữ liệu giữ riêng để chấm điểm, không dùng để học. |
| **Top-k accuracy** | Tỉ lệ đáp án nằm trong k lựa chọn hàng đầu của mô hình. |
| **Checkpoint** | File lưu bộ tham số (`gomoku_policy.pt`). |
| **Policy network** | Mạng trả lời "nên đi đâu" dưới dạng xác suất cho mỗi nước. |
| **Mixed precision** | Dùng số 16-bit cho phần lớn phép tính để chạy nhanh hơn. |
