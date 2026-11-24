# Tự động lưu ván cờ từ trang web vào Supabase

Trang web ([docs/index.html](../docs/index.html)) tự gửi mỗi ván vào một bảng Supabase khi ván kết thúc. Ván bỏ dở từ 10 nước trở lên cũng được gửi khi người chơi bấm "Ván mới" hoặc đóng trang. Chỉ lưu nước đi, kết quả, quân người chơi cầm, độ khó và số lần đi lại; không lưu thông tin cá nhân. Người chơi có thể bỏ chọn ô "Lưu ván ẩn danh" trên trang để không gửi.

## Cài đặt một lần

1. Vào https://supabase.com, đăng nhập (có thể dùng tài khoản GitHub) rồi bấm **New project**. Đặt tên (ví dụ `gomoku-agent`), chọn region gần bạn (ví dụ Singapore), đặt mật khẩu database và lưu nó lại ở chỗ riêng (không cần cho trang web).
2. Khi project tạo xong, mở **SQL Editor → New query**, dán toàn bộ nội dung file [games.sql](games.sql), bấm **Run**. Lệnh này tạo bảng `games` và chặn quyền: khoá công khai chỉ được **thêm** ván, không đọc, sửa hay xoá được.
3. Mở **Project Settings → API Keys** (hoặc **Connect**) và lấy:
   - **Project URL**, dạng `https://abcdxyz.supabase.co`
   - **Publishable key** (`sb_publishable_...`), hoặc **anon public key** (chuỗi dài bắt đầu bằng `eyJ`) nếu project dùng kiểu khoá cũ
4. Điền hai giá trị đó vào đầu phần script trong [docs/index.html](../docs/index.html):
   ```js
   const SUPABASE_URL = 'https://abcdxyz.supabase.co';
   const SUPABASE_KEY = 'sb_publishable_...';
   ```
   rồi commit và push. Khi hai giá trị còn trống, trang web không gửi gì và ẩn ô lưu ván.

Publishable/anon key được thiết kế để nằm công khai trong trang web; bảo vệ dữ liệu là nhờ các quy tắc trong `games.sql`.
**Tuyệt đối không** đưa **secret key** (`sb_secret_...`) hay **service_role key** vào trang web hoặc repo: khoá đó có toàn quyền với database.

## Xem và tải ván về

- Xem nhanh: **Table Editor → games** trong dashboard Supabase.
- Tải về để train: lấy **secret key** (Project Settings → API Keys), chỉ đặt trong biến môi trường trên máy bạn, rồi chạy:
  ```powershell
  $env:SUPABASE_URL = "https://abcdxyz.supabase.co"
  $env:SUPABASE_SECRET_KEY = "sb_secret_..."
  python fetch_games.py --source supabase
  ```
  Các ván được ghi vào `data/human_games.jsonl`. Một ván có thể được lưu nhiều lần (ví dụ người chơi đi lại sau khi ván kết thúc), các bản đó có cùng `game_uid`; script giữ bản đã kết thúc mới nhất, hoặc bản mới nhất nếu ván chưa từng kết thúc. Ván có nước trùng ô hoặc kết quả không khớp với bàn cờ bị bỏ.

## Giới hạn

- Bảng kiểm tra định dạng (nước đi hợp lệ, tối đa 225 nước, kết quả thuộc danh sách cho phép), nhưng không chặn được người cố tình gửi ván giả. Khi train nên lọc lại, ví dụ bỏ ván có nước đi trùng ô hoặc kết quả không khớp với bàn cờ (`fetch_games.py` đã làm việc này).
- Gói miễn phí của Supabase có 500 MB database, đủ cho hàng triệu ván. Project miễn phí bị tạm dừng nếu không có hoạt động trong khoảng 1 tuần; khi đó vào dashboard bấm Restore.
