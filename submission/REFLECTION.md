# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Ngạc nhiên nhất là `attn_only` — chỉ gắn LoRA vào 2 module (q, v) thay vì toàn bộ 12
module tuyến tính — lại **hoà tuyệt đối** với `correct` trên target (0.970 cả hai), dù
train loss của nó còn thấp hơn (0.537 so với 0.6264). Mình từng nghĩ "gắn LoRA đúng vị
trí" là yếu tố quyết định, nhưng số liệu cho thấy một khi rank được nâng lên để bù đủ
tham số (16 → 283), vị trí gắn gần như không còn quan trọng. Ngạc nhiên thứ hai là độ
chênh giữa hai cách nhìn kết quả: nếu chỉ nhìn `target` thì fine-tune trông như thành
công rực rỡ (+0.205), nhưng `regression` tụt tới -0.213 — gấp hơn 10 lần ngưỡng cho phép.
Một con số đẹp duy nhất đã suýt khiến mình kết luận sai hoàn toàn.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Phần tốn thời gian nhất không phải là debug code (vì toàn bộ `notebooks/*.py` đã được
viết sẵn, không có stub) mà là **thao tác hạ tầng trên Kaggle**: bật GPU/Internet bị
khoá do chưa xác minh số điện thoại, lỗi `ConcurrencyViolation` khi lưu draft, và việc
gom toàn bộ 8 bước vào một lần "Save & Run All (Commit)" để không phải ngồi canh ~2 giờ
chạy pipeline. Đây không phải chỗ mình dự đoán trước — mình nghĩ phần khó sẽ là đọc hiểu
và diễn giải kết quả (mask proof, loss, verdict), nhưng thực ra phần đó khá rõ ràng nhờ
rubric chỉ dẫn từng bước; phần mất thời gian thật sự lại là những trục trặc vặt của nền
tảng chạy (Kaggle) chứ không phải nội dung chuyên môn của lab.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Trước đây mình tin rằng cứ fine-tune ra `target` cao hơn baseline là coi như thành công,
và loss thấp hơn đồng nghĩa với cấu hình tốt hơn. Sau khi thấy `wrong_lr` (chỉ đổi đúng 1
con số LR) khiến model gần như không học được gì (`target=0.000`) dù loss vẫn giảm nhẹ,
và thấy chính fine-tune "thắng" của mình fail cổng hồi quy vì quên mất 21 điểm phần trăm
khả năng trả lời câu hỏi chung — mình không còn tin vào việc đánh giá fine-tune chỉ bằng
một con số duy nhất nữa. Một mô hình có thể đồng thời đúng và sai, tuỳ vào việc bạn đo
cái gì.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Mình dùng Claude để: đọc toàn bộ repo và tóm tắt rubric/cấu trúc trước khi bắt đầu, viết
`plan.md` theo từng giai đoạn, viết cell bootstrap cho Kaggle (set env var, pin
`torchao`), và sau khi chạy xong pipeline — trích xuất số liệu từ output notebook để điền
`REPORT.md` (bảng baseline, bảng misconfig autopsy, phân tích 3 câu hỏi NB4, diễn giải
verdict, bảng định tính có đối chiếu nhãn thật từ `data/eval_target.jsonl`). Chỗ cần
chỉnh lại: khi mới điền §3, AI dùng số liệu từ lần chạy NB2 **đầu tiên** (cell 4) trong
khi notebook lỡ chạy NB2 **hai lần** liên tiếp — phải soát lại và thay bằng số liệu của
lần chạy cuối (vì đó mới là nội dung thực sự ghi vào `baselines_frozen.json`). Ngoài ra
AI không thể tự điền Họ tên/MSSV và không thể viết phần Reflection này một cách trung
thực — đây là giới hạn hợp lý, vì những phần đó cần chính người làm xác nhận.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Bước đầu tiên sẽ không phải là cấu hình LoRA, mà là **kiểm tra loss mask** giống NB1 —
decode lại đoạn được tính loss để chắc chắn mô hình học đúng câu trả lời chứ không phải
học lại câu hỏi. Sau đó, dựa trên bài học từ lab này, bước tiếp theo là đảm bảo tập train
có trộn một phần nhỏ (1-5%) dữ liệu tổng quát để tránh catastrophic forgetting ngay từ
đầu, thay vì đợi đo regression xong mới phát hiện ra vấn đề.
