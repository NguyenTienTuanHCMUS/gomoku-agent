# Gomoku Agent

Một AI chơi cờ caro (Gomoku freestyle, bàn 15×15, 5 quân liền là thắng). Mạng nơ-ron học bắt chước các ván của giải [Gomocup 2019](https://gomocup.org/results/), rồi được ghép với tìm kiếm cây Monte Carlo (MCTS) giống AlphaZero.

**Chơi thử ngay trên trình duyệt:** https://nguyentientuanhcmus.github.io/gomoku-agent/
Model chạy hoàn toàn trên máy người chơi, không cần cài gì.

**Đọc cách nó hoạt động:** [trang giải thích có hình minh hoạ](https://nguyentientuanhcmus.github.io/gomoku-agent/explainer.html) · [GIAI_THICH.md](GIAI_THICH.md) (bản chi tiết có công thức toán, có cả [PDF](GIAI_THICH.pdf))

## Kết quả

| Thành phần | Kết quả (đo trên các ván chưa từng dùng để train) |
|---|---|
| Policy: đoán nước tiếp theo của engine Gomocup | top-1 **54,7%**, top-5 **86%** |
| Value: đoán ai sẽ thắng từ thế cờ hiện tại | đúng **76,5%** (65,6% đầu ván, 82,6% khi bàn đã có 30+ quân) |

Mạng: ResNet 8 block × 96 kênh, 1,34 triệu tham số (policy), cộng 7.361 tham số cho value head.

## Cách hoạt động, tóm tắt

1. **Dữ liệu** ([create_dataset.py](create_dataset.py)): 10.224 ván Freestyle của Gomocup 2019 vốn chơi trên bàn 20×20. Ván nào có toàn bộ quân nằm gọn trong khung 15×15 thì được dời vào bàn 15×15 (giữ được 9.246 ván, 321.774 thế cờ). Mỗi mẫu gồm bàn cờ (nhìn từ phía người sắp đi), nước engine đã đi, và kết quả ván.
2. **Policy** ([train_policy.ipynb](train_policy.ipynb)): bài toán phân loại 225 lớp, train trên GPU T4 của Kaggle khoảng 23 phút, có tăng cường dữ liệu bằng 8 phép đối xứng của bàn cờ.
3. **Value head** ([train_value.py](train_value.py)): một đầu ra nhỏ gắn lên phần thân của policy (giữ nguyên phần thân), dự đoán kết quả ván. Train trên CPU khoảng 20 phút.
4. **MCTS** ([gomoku.py](gomoku.py)): PUCT như AlphaZero, cộng thêm kiểm tra chiến thuật chính xác ở mỗi nút: có nước thắng ngay thì đi, đối thủ có hai đường thắng thì coi là thua, đối thủ có một đường thắng thì buộc phải chặn.
5. **Web** ([docs/](docs)): model được xuất sang ONNX và chạy bằng ONNX Runtime Web trong một Web Worker; MCTS được viết lại bằng JavaScript và giữ lại cây tìm kiếm giữa các lượt.

## Chạy trên máy

```powershell
python -m pip install numpy torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
python play.py                 # đánh với model trong terminal: bạn cầm X, model tìm kiếm 400 lượt mỗi nước
python play.py --ai-first      # model cầm X
python play.py --sims 0        # không tìm kiếm: mạng + kiểm tra thắng/chặn
```

Nhập nước đi kiểu `h8` (ô giữa bàn). Các lệnh: `hint`, `undo`, `quit`.

Các script khác:

| Lệnh | Việc làm |
|---|---|
| `python create_dataset.py` | Tải file zip Gomocup 2019 và tạo `data/gomoku_15x15.npz` (thêm `--winner-only` để chỉ lấy nước của người thắng) |
| `train_policy.ipynb` | Train policy trên Kaggle: upload file `.npz` thành Dataset, chọn GPU T4, Run All, tải `gomoku_policy.pt` về `models/run1/` |
| `python train_value.py` | Train value head trên CPU → `models/pv1/gomoku_pv.pt` |
| `python arena.py --a models/pv1/gomoku_pv.pt:200 --b models/run1/gomoku_policy.pt:0` | Cho hai agent đấu với nhau từ các khai cuộc Gomocup, mỗi khai cuộc 2 ván đổi màu (`:0` nghĩa là không tìm kiếm) |
| `python export_onnx.py` | Xuất model sang `docs/model.onnx` cho trang web |
| `python -m http.server 8765 --directory docs` | Chạy thử trang web trên máy tại http://localhost:8765 |
| `python fetch_games.py` | Tải các ván người chơi đã gửi (GitHub Issues) về `data/human_games.jsonl` |

## Gửi ván cờ để model học thêm

Sau mỗi ván trên trang web có nút **Gửi ván này lên GitHub**. Nút này mở trang tạo Issue với nội dung ván cờ điền sẵn; bạn xem lại rồi bấm Submit (cần tài khoản GitHub, và Issue sẽ công khai). Những ván người thắng model đặc biệt có giá trị, vì chúng chỉ ra chỗ model còn yếu.

## Cấu trúc thư mục

```
create_dataset.py   tạo dataset từ ván Gomocup
train_policy.ipynb  train policy (Kaggle GPU)
train_value.py      train value head (CPU)
gomoku.py           mạng, luật cờ, MCTS
play.py             chơi trong terminal
arena.py            cho các agent đấu với nhau
export_onnx.py      xuất model cho web
fetch_games.py      tải ván người chơi gửi lên
data/               dataset đã xử lý (.npz)
models/run1/        policy (kèm biểu đồ train từ Kaggle)
models/pv1/         policy + value head
docs/               trang web (GitHub Pages): index.html, engine-worker.js, model.onnx, explainer.html
```

## Nguồn

- Ván cờ: kết quả giải [Gomocup 2019](https://gomocup.org/results/gomocup-result-2019/) (hạng Freestyle).
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) 1.20.1 (MIT), đặt trong `docs/vendor/ort/`.
