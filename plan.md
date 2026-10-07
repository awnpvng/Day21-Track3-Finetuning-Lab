# Plan hoàn thành Lab 21 (Fine-tuning & Safety) — chạy trên Kaggle

Quy ước nhãn mỗi bước: `[hoạt động][người thực hiện]`

- Hoạt động: `[run]` chạy lệnh/notebook, `[report]` viết nội dung báo cáo, `[code]` sửa/viết code
- Người thực hiện: `[AI]` Claude làm, `[manual]` bạn tự làm trên Kaggle (vì code chỉ sửa ở máy, chỉ **chạy** trên Kaggle)

Ghi chú quan trọng đã xác nhận từ repo:

- Toàn bộ `notebooks/01..06_*.py` **đã code đầy đủ**, không có stub cần điền → việc chính là **chạy đúng pipeline + viết báo cáo**, không phải viết code pipeline từ đầu.
- Tier dùng cho Kaggle T4×2: `COMPUTE_TIER=T4`, chỉ dùng **1 GPU** (lab không cần multi-GPU).
- `.env` không tự được Kaggle đọc — phải set qua `os.environ` ở cell đầu notebook trên Kaggle, hoặc gọi `labkit.env.load_dotenv(override=True)` sau khi tạo file `.env`.
- `scripts/verify.py` chính là bộ rubric hoá thành check tự động — chạy nó trước khi nộp là bắt buộc.

---

## Giai đoạn 0 — Chuẩn bị ở máy local (code/chỉnh sửa, không chạy)

- [X] `[code][AI]` Đọc lại `README.md`, `rubric.md`, `HARDWARE-GUIDE.md` để chốt lựa chọn: tier=`T4`, model=`Qwen3.5-4B` (default), dataset=250 ticket có sẵn (không đổi — tránh phức tạp thêm `CUSTOM_DATASET.md` trừ khi làm bonus B2).
- [X] `[code][AI]` Tạo `.env` từ `.env.example` với `COMPUTE_TIER=T4` (giữ nguyên `MASK_MODE=assistant-only`, `EPOCHS=2`, không set `EVAL_LIMIT` để tính điểm đầy đủ).
- [X] `[code][AI]` Viết 1 cell "Kaggle bootstrap" (đầu mỗi notebook khi chạy trên Kaggle) để: clone/sync repo vào `/kaggle/working`, `pip install -r requirements.txt` (bỏ qua torch vì Kaggle có sẵn), set `os.environ["COMPUTE_TIER"]="T4"` trước khi `import labkit`, xử lý pin `torchao>=0.16` (Kaggle có thể có bản cũ gây lỗi `get_peft_model()`).
- [X] `[code][AI]` Kiểm tra `requirements-cpu.txt`/`requirements.txt` không xung đột với package Kaggle preinstall (torch, bitsandbytes Linux-only — Kaggle GPU image là Linux nên OK).
- [X] `[report][manual]` Quyết định cuối: giữ model/dataset mặc định hay đổi (nếu đổi, phải chạy `scripts/check_mask_agreement.py` lại NB1 và ghi lý do vào báo cáo).

## Giai đoạn 1 — Upload & khởi tạo trên Kaggle

- [X] `[run][manual]` Tạo Kaggle Notebook mới, Accelerator = **GPU T4 x2** (chỉ dùng 1 GPU theo hướng dẫn), Internet = On.
- [X] `[run][manual]` Push code lên GitHub (hoặc upload làm Kaggle Dataset) rồi `git clone` vào notebook Kaggle; hoặc add repo as Kaggle Dataset input.
- [X] `[run][manual]` Chạy cell bootstrap (`pip install -r requirements.txt`, set env vars) — xác nhận `import labkit` không lỗi.
- [X] `[run][manual]` Chạy `make smoke` (hoặc `pytest tests/` nếu Makefile không chạy tốt trong Kaggle shell) để xác nhận môi trường OK trước khi chạy pipeline dài.

## Giai đoạn 2 — Chạy pipeline chính (NB1 → NB5)

- [X] `[run][manual]` **NB1** `01_data_and_mask.py` (~1 phút, CPU-ok) → tạo `results/mask_proof.json`, `template_check.json`, `token_stats.json`.
- [X] `[report][manual]` Kiểm tra `mask_proof.json`: cả 2 assert (answer trong loss, question KHÔNG trong loss) phải green, và `supervised_fraction < 0.95`. Nếu fail → debug trước khi tiếp tục (đây là gốc của toàn bộ điểm pipeline).
  - Kết quả: `answer_is_supervised=true`, `question_is_masked=true`, `supervised_fraction=0.4149` (< 0.95) → ĐẠT.
  - ⚠️ `token_stats.json`: p95=98, tier T4 mặc định `max_length=1024` → lệch nhiều, cần giải thích lý do giữ 1024 trong REPORT.md §1 (mục rubric 1.3).
  - Split: train=225 / val=25 (seed 42).
- [X] `[run][manual]` **NB2** `02_baselines.py` (~17-23 phút, GPU) → đóng băng `baselines_frozen.json`. Xác nhận baseline (b) > (a).
  - Kết quả: (a) target=0.0, format=0.0, regression=0.758, latency=2868ms — naive prompt không ra JSON hợp lệ.
  - (b) target=0.765, format=1.0, regression=0.758, latency=931ms — (b) >> (a) → ĐẠT rubric 3.1.
  - `optimized_prompt_sha=719e74d3b6232053` đã đóng băng, không sửa `OPTIMIZED_PROMPT` sau bước này.
- [X] `[run][manual]` **NB3** `03_train_correct.py` (~15-25 phút, GPU) → `adapters/correct/` + dòng `correct` trong `runs.csv`.
  - Kết quả: r=16, trainable=32,464,896, lr=1e-4, 30 bước (2 epoch), final_loss=0.6264, train=827.6s, peak VRAM=8.78GB.
- [X] `[run][manual]` **NB4** `04_misconfig_autopsy.py` (~45-60 phút, GPU) → 3 run đối chứng (`attn_only`, `wrong_lr`, `qlora`) cùng `max_steps` với `correct`.
  - `attn_only`: r=283 (matched param), trainable=32,456,704 (lệch 0.025% so với `correct` → đạt rubric 2.1), final_loss=0.537, VRAM=8.79GB.
  - `wrong_lr`: lr=1e-5 (giảm 10x), final_loss=1.5705 — cao hẳn, không hội tụ.
  - `qlora`: 4-bit, final_loss=0.7058, VRAM chỉ 3.86GB (tiết kiệm ~56% so với `correct`).
  - Cả 4 run đều đúng 30 bước → đạt rubric 2.2.
- [X] `[run][manual]` **NB5** `05_evaluate_and_verdict.py` (~21 phút, GPU) → `verdict.json`, `autopsy.json`, `qualitative.json` (đủ 4 nhóm: target/regression/format/latency).
  - (c) fine-tune: target=0.97, regression=0.544, format=1.0, latency=1284ms.
  - **Verdict: FAILED** — target Δ=+0.205 (vượt baseline b) nhưng regression Δ=−0.213 (vượt ngưỡng cho phép 0.020) → catastrophic forgetting rõ rệt.
  - Target theo từng run (NB5 §4, dùng để xếp hạng thay vì final_loss NB4): `correct`=0.97, `attn_only`=0.97 (hoà!), `wrong_lr`=0.0, `qlora`=0.94.
  - 3 ca target tệ nhất đều sai ở field `urgency` (dự đoán `trung_binh` thay vì đúng `thap`) khi khách dùng câu lịch sự ("khi nào tiện", "cảm ơn shop nhiều") — pattern hệ thống.
- [X] `[run][manual]` (Bonus B1 — đã làm) **NB6** `06_merge_and_serve.py` (~10 phút) → `merge_check.json`: trước merge 0.9700, sau merge 0.9700 (Δ=0.0000, không tụt điểm) + hot-swap thành công 3 adapter (`correct`, `attn_only`, `qlora`) trên cùng 1 base.
- [X] `[run][manual]` Chạy `python scripts/verify.py` trên Kaggle → 25 passed / 1 warning / **1 FAIL duy nhất**: REPORT.md còn 6 placeholder (`<điền>`, `<paste>`...) chưa điền — mọi check kỹ thuật khác đều PASS (mask proof, baseline b>a, 1 step budget, attn_only fair contrast, verdict recorded).
- [X] `[run][manual]` Lấy kết quả về máy local: Kaggle tự động push `lab21.ipynb` (có đầy đủ output) lên GitHub qua tính năng tích hợp → đã `git pull` về, trích xuất toàn bộ số liệu từ notebook output thay vì cần tải riêng `results/`.
  - ⚠️ Lưu ý còn thiếu: `adapters/correct/` (file .safetensors) và các file json gốc trong `results/` **chưa có trong repo local** — notebook chỉ mang theo phần text output, không phải file nhị phân. Cần tải riêng từ tab "Output" của Kaggle version đã commit trước khi đóng gói nộp bài (xem Giai đoạn 5).

## Giai đoạn 3 — Viết báo cáo (ở máy local, dựa trên kết quả từ Kaggle)

- [X] `[report][AI]` Điền `submission/REPORT.md` dựa trên output đầy đủ từ `lab21.ipynb` (Kaggle tự push lên GitHub): §1 Setup, §2 Mask proof, §3 Ba baseline, §4 Misconfig autopsy (bảng + 3 câu trả lời ≥3 câu), §5 Phán quyết (FAILED, diễn giải nguyên nhân), §6 Định tính (6 ví dụ: 3 thắng + 3 thua, nhãn đúng lấy từ `data/eval_target.jsonl`, phát hiện pattern lỗi `urgency`), Phụ lục (tick B1 đã làm qua NB6).
- [X] `[report][AI]` Viết kết luận + 3 lessons learned + "2 giờ nữa sẽ thử gì" dựa trên số liệu thật.
- [X] `[report][AI]` Đối chiếu số liệu: mọi số trong REPORT.md lấy trực tiếp từ output notebook đã chạy (không bịa), kể cả việc ghi chú minh bạch khi NB2 bị chạy trùng 2 lần.
- [ ] `[report][manual]` **Còn thiếu**: điền **Họ tên** và **MSSV** ở đầu `REPORT.md` (2 placeholder cuối cùng, `verify.py` sẽ FAIL cho đến khi điền).
- [ ] `[report][manual]` Điền `submission/REFLECTION.md` (5 câu hỏi phản tư, viết cá nhân/cụ thể, tránh generic — phần này cần trải nghiệm cá nhân của bạn, AI không thay được).

## Giai đoạn 4 — Bonus (tuỳ chọn, làm thêm trên Kaggle nếu muốn +15đ)

- [ ] `[run][manual]` B1 (+3): đã làm ở NB6 nếu chạy.
- [ ] `[code+run][manual]` B2 (+3): chuẩn bị dataset riêng ≥200 mẫu + viết `data/CUSTOM_DATASET.md` (nguồn, kích thước, decontamination), rồi chạy lại toàn bộ pipeline NB1-5 với dataset mới.
- [ ] `[run][manual]` B3 (+4): train lại NB3 hai lần với `MASK_MODE=assistant-only` và `response-only` trên model có khả năng reasoning, report `valid_trace_rate`.
- [ ] `[run][manual]` B4 (+3): sweep rank r∈{8,16,64} với `placement=text-linear` cố định, phân tích khi nào rank là lever chính.
- [ ] `[run][manual]` B5 (+2): push adapter lên HuggingFace Hub (public) + dẫn link trong report (cần `HF_TOKEN` trong `.env`).

## Giai đoạn 5 — Verify & nộp bài

- [ ] `[run][manual]` Chạy `python scripts/verify.py` (trên Kaggle hoặc local sau khi copy `results/` + `adapters/correct/` về) — phải exit code 0, không còn placeholder `<điền>`/`<paste>`/`<0.xx>` trong REPORT.md, report ≥400 từ.
- [ ] `[code][AI]` Nếu `verify.py` báo FAIL ở mục nào, Claude hỗ trợ xác định nguyên nhân (đọc file kết quả liên quan) để bạn sửa đúng chỗ.
- [ ] `[run][manual]` Đóng gói theo 1 trong 3 format (`submission/REPORT.md` + `results/` + `adapters/correct/` + `notebooks/` đã clear output) — khuyến nghị Option A (ZIP gọn).
- [ ] `[run][manual]` Nộp bài.

---

### Lưu ý rủi ro khi chạy trên Kaggle (theo HARDWARE-GUIDE.md / SIMULATION-FINDINGS.md)

- Kaggle session có giới hạn thời gian (~9-12h/tuần GPU quota, mỗi session tối đa 9h chạy liên tục) — pipeline NB1-5 tốn 100-130 phút nên nằm trong giới hạn 1 session, nhưng nên lưu kết quả `results/` sau mỗi notebook để tránh mất khi session bị ngắt.
- T4 không hỗ trợ bf16 thật (Turing) — `labkit/device.py` tự fallback fp16, không cần chỉnh gì.
- Nếu OOM ở NB4 run thứ 2: đảm bảo `generate.free_memory()` được gọi giữa các run (đã có trong code, chỉ cần không sửa).
- torchao version trên Kaggle image có thể cũ hơn yêu cầu — cell bootstrap cần `pip install -U "torchao>=0.16"` trước khi train.
