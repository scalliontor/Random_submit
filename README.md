# Bot điền Google Form

## Điền dữ liệu người tham gia

`form_bot.py` nhận CSV hoặc Google Sheet công khai. Mẫu `responses.csv` khớp 35 trường của khảo sát đã cấu hình.

Xem trước dữ liệu trước khi gửi:

```bash
python3 form_bot.py fill 'https://docs.google.com/forms/d/e/1FAIpQLScw4GlKtaVAp7LOfM9IqI79KRhxbe2oD0td0aLlvDCRQhMzWw/viewform' responses.csv --dry-run
```

Bỏ `--dry-run` sau khi đã kiểm tra câu trả lời. Bot lưu trạng thái gửi trong `.form_bot_state.json` để tránh gửi lại các dòng đã hoàn tất.

## Benchmark Gemma theo persona

`gemma_form_bot.py` đọc câu hỏi và lựa chọn trực tiếp từ Form, gọi API OpenAI-compatible của Gemma, kiểm tra câu trả lời rồi lưu CSV và checkpoint cục bộ. Nếu có `--submit`, bot gửi 10 phản hồi mỗi đợt, nghỉ ngẫu nhiên 2–3 giờ giữa các đợt và tiếp tục từ checkpoint nếu bị dừng. Mặc định bot chỉ tạo CSV, không gửi lên Form.

Theo yêu cầu, Form không được thêm cột phân biệt người thật và persona Gemma. Vì vậy file tải trực tiếp từ Google Forms cũng không có dấu nguồn; checkpoint `.gemma_form_state.json` lưu dấu vết mô phỏng ở máy chạy bot.

Form hiện tại không hỏi tên; persona chỉ tạo tuổi, phân khu, giới tính, học vấn, tình trạng cư trú và câu trả lời nhất quán theo tính cách. Hướng dẫn Gemma ưu tiên thang điểm 3–5, cho phép một số ít 1–2 khi hợp lý và tránh mẫu điểm máy móc.

### Kết nối Gemma qua endpoint IP tĩnh

Bot tự đọc `GEMMA_API_KEY` từ file `.env` cạnh mã nguồn; biến môi trường cùng tên, nếu đã đặt, sẽ được ưu tiên. `.env` đã được thêm vào `.gitignore`. Endpoint mặc định lấy theo tài liệu Gemma: `http://171.226.10.121:8000/llm/v1/`. Không cần SSH tunnel.

Nếu muốn đổi endpoint hoặc khóa tạm thời, có thể đặt biến môi trường:

```bash
export GEMMA_API_KEY='điền-key-của-bạn'
export GEMMA_BASE_URL='http://171.226.10.121:8000/llm/v1/'
```

Endpoint tài liệu dùng HTTP, không mã hóa khóa Bearer khi truyền qua mạng. Chỉ dùng trên mạng bạn tin cậy; nếu máy chủ có HTTPS, hãy đổi `GEMMA_BASE_URL` sang địa chỉ HTTPS. Có thể đặt model khác bằng `GEMMA_MODEL`.

### Tạo CSV để xem trước

```bash
python3 gemma_form_bot.py 'https://docs.google.com/forms/d/e/1FAIpQLScw4GlKtaVAp7LOfM9IqI79KRhxbe2oD0td0aLlvDCRQhMzWw/viewform' --count 50 --batch-size 10
```

Lệnh trên tạo `gemma_responses.csv` và `.gemma_form_state.json`, chưa gửi phản hồi. Mở CSV để xem câu trả lời trước.

### Tạo và gửi theo từng đợt

```bash
python3 gemma_form_bot.py 'https://docs.google.com/forms/d/e/1FAIpQLScw4GlKtaVAp7LOfM9IqI79KRhxbe2oD0td0aLlvDCRQhMzWw/viewform' --count 50 --batch-size 10 --submit
```

Lệnh này gửi các phản hồi đã sinh hoặc sinh thêm từng đợt, nghỉ 2–3 giờ giữa các đợt. Giữ máy và mạng hoạt động trong lúc chạy. Dừng bằng `Ctrl+C`; chạy lại đúng lệnh và giữ nguyên `.gemma_form_state.json` để tiếp tục. Nếu kết quả gửi không rõ, bot dừng và yêu cầu kiểm tra Google Forms trước khi thử lại để tránh trùng phản hồi.

Có thể đổi nhịp, ví dụ `--min-pause-hours 2 --max-pause-hours 3`. Cần Python 3.10 trở lên; không cần cài package ngoài.
