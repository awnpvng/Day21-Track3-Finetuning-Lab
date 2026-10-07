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

- [ ] `[code][AI]` Đọc lại `README.md`, `rubric.md`, `HARDWARE-GUIDE.md` để chốt lựa chọn: tier=`T4`, model=`Qwen3.5-4B` (default), dataset=250 ticket có sẵn (không đổi — tránh phức tạp thêm `CUSTOM_DATASET.md` trừ khi làm bonus B2).
- [ ] `[code][AI]` Tạo `.env` từ `.env.example` với `COMPUTE_TIER=T4` (giữ nguyên `MASK_MODE=assistant-only`, `EPOCHS=2`, không set `EVAL_LIMIT` để tính điểm đầy đủ).
- [ ] `[code][AI]` Viết 1 cell "Kaggle bootstrap" (đầu mỗi notebook khi chạy trên Kaggle) để: clone/sync repo vào `/kaggle/working`, `pip install -r requirements.txt` (bỏ qua torch vì Kaggle có sẵn), set `os.environ["COMPUTE_TIER"]="T4"` trước khi `import labkit`, xử lý pin `torchao>=0.16` (Kaggle có thể có bản cũ gây lỗi `get_peft_model()`).
- [ ] `[code][AI]` Kiểm tra `requirements-cpu.txt`/`requirements.txt` không xung đột với package Kaggle preinstall (torch, bitsandbytes Linux-only — Kaggle GPU image là Linux nên OK).
- [ ] `[report][manual]` Quyết định cuối: giữ model/dataset mặc định hay đổi (nếu đổi, phải chạy `scripts/check_mask_agreement.py` lại NB1 và ghi lý do vào báo cáo).

## Giai đoạn 1 — Upload & khởi tạo trên Kaggle

- [ ] `[run][manual]` Tạo Kaggle Notebook mới, Accelerator = **GPU T4 x2** (chỉ dùng 1 GPU theo hướng dẫn), Internet = On.
- [ ] `[run][manual]` Push code lên GitHub (hoặc upload làm Kaggle Dataset) rồi `git clone` vào notebook Kaggle; hoặc add repo as Kaggle Dataset input.
- [ ] `[run][manual]` Chạy cell bootstrap (`pip install -r requirements.txt`, set env vars) — xác nhận `import labkit` không lỗi.
- [ ] `[run][manual]` Chạy `make smoke` (hoặc `pytest tests/` nếu Makefile không chạy tốt trong Kaggle shell) để xác nhận môi trường OK trước khi chạy pipeline dài.

## Giai đoạn 2 — Chạy pipeline chính (NB1 → NB5)

- [ ] `[run][manual]` **NB1** `01_data_and_mask.py` (~1 phút, CPU-ok) → tạo `results/mask_proof.json`, `template_check.json`, `token_stats.json`.
- [ ] `[report][manual]` Kiểm tra `mask_proof.json`: cả 2 assert (answer trong loss, question KHÔNG trong loss) phải green, và `supervised_fraction < 0.95`. Nếu fail → debug trước khi tiếp tục (đây là gốc của toàn bộ điểm pipeline).
- [ ] `[run][manual]` **NB2** `02_baselines.py` (~17-23 phút, GPU) → đóng băng `baselines_frozen.json`. Xác nhận baseline (b) > (a).
- [ ] `[run][manual]` **NB3** `03_train_correct.py` (~15-25 phút, GPU) → `adapters/correct/` + dòng `correct` trong `runs.csv`.
- [ ] `[run][manual]` **NB4** `04_misconfig_autopsy.py` (~45-60 phút, GPU) → 3 run đối chứng (`attn_only`, `wrong_lr`, `qlora`) cùng `max_steps` với `correct`. Nếu Kaggle session timeout/crash giữa chừng, dùng `FORCE_RETRAIN=1`/`ONLY=<run>` để resume, không chạy lại từ đầu.
- [ ] `[run][manual]` **NB5** `05_evaluate_and_verdict.py` (~21 phút, GPU) → `verdict.json`, `autopsy.json`, `qualitative.json` (đủ 4 nhóm: target/regression/format/latency).
- [ ] `[run][manual]` (Tuỳ chọn, bonus B1) **NB6** `06_merge_and_serve.py` (~10 phút) → `merge_check.json`, hot-swap ≥2 adapter.
- [ ] `[run][manual]` Download toàn bộ `results/*.json`, `results/runs.csv`, `adapters/correct/` từ Kaggle output về máy local (qua "Save Version" + tab Output, hoặc zip rồi tải).

## Giai đoạn 3 — Viết báo cáo (ở máy local, dựa trên kết quả từ Kaggle)

- [ ] `[report][manual]` Điền `submission/REPORT.md`: model+dataset+lý do chọn, bằng chứng mask (từ `mask_proof.json`), bảng 3 baseline (NB2), bảng misconfig autopsy (NB4) + trả lời 3 câu hỏi ≥3 câu mỗi câu, verdict NB5 (đọc diễn giải ≥100 từ, PASS hoặc FAIL đều được tính điểm nếu lập luận đúng), bảng qualitative ≥5 ví dụ trong đó ≥2 ví dụ fine-tune THUA (không được cherry-pick).
- [ ] `[report][manual]` Viết phần kết luận ≥150 từ có lập luận nhân-quả (không chỉ liệt kê số), + 3 lessons learned cụ thể + "nếu có thêm 2 giờ sẽ làm gì".
- [ ] `[report][AI]` Rà soát: mọi số trong `REPORT.md` khớp đúng với file trong `results/` (yêu cầu 4.3, 5 điểm) — Claude có thể hỗ trợ đối chiếu số liệu.
- [ ] `[report][manual]` Điền `submission/REFLECTION.md` (5 câu hỏi phản tư, viết cá nhân/cụ thể, tránh generic).

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
