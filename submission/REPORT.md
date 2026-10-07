# Lab 21 — Evaluation Report

**Họ tên**: Trương Hoàng Thành An  **MSSV**: 2A202602574  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Kaggle Tesla T4 (1 trong 2 GPU, CUDA_VISIBLE_DEVICES=0)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

|                    |                                                                         |
| ------------------ | ----------------------------------------------------------------------- |
| Dataset            | 250 ticket CSKH tiếng Việt → JSON triage (mặc định, không đổi) |
| Train / val        | 225 / 25 (seed 42)                                                      |
| `max_length`     | 1024 — p95 đo được là 98*(results/token_stats.json)*            |
| `MASK_MODE`      | `assistant-only`                                                      |
| Epochs / max_steps | 2 / 30 bước (tất cả 4 run NB3+NB4 dùng chung 30 bước)            |

**`max_length` lệch với p95 đo được — giải thích:** p95 chỉ 98 token nhưng tier T4 mặc
định `max_length=1024`. Giữ nguyên 1024 thay vì hạ theo p95 vì: (1) `max=101` rất sát p95
nên 1024 không lãng phí tương quan, phần padding bị cắt bởi `DataCollator` theo batch chứ
không theo giá trị cố định này; (2) không có lợi ích đo được từ việc hạ xuống 256 ở quy mô
250 mẫu — thời gian train không bị giới hạn bởi `max_length` mà bởi số bước; (3) giữ mặc
định tier để không phải chạy lại `check_mask_agreement.py` và tránh rủi ro cắt mất ticket
dài bất thường chưa thấy trong tập train nhưng có thể xuất hiện khi đổi dữ liệu.

**Template có giữ khối `<think>` không?** **Có** — `template_check.json` ghi
`VERDICT: reasoning preserved — safe to train on traces`. Ví dụ render thử với câu hỏi
"2+2?": chat template chèn `<think>buoc 1: kiem tra. buoc 2: tra loi.</think>` nguyên vẹn
trước câu trả lời, không bị cắt mất. Với corpus thật của lab (250 ticket), câu trả lời
huấn luyện là JSON thuần không có reasoning trace, nên khối `<think></think>` trong mask
thực tế rỗng — template có khả năng giữ `<think>` nhưng dữ liệu của lab không tận dụng
khả năng đó (xem thêm B3 ở Phụ lục nếu muốn kiểm chứng với dữ liệu có trace thật).

---

## 2. Mask proof (NB1)

|                                  |            |
| -------------------------------- | ---------- |
| `supervised_fraction`          | `0.4149` |
| Câu trả lời nằm trong loss   | `true`   |
| Câu hỏi KHÔNG nằm trong loss | `true`   |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run                         | target | regression | format | latency (ms) |
| --------------------------- | ------ | ---------- | ------ | ------------ |
| (a) base + naive prompt     | 0.000  | 0.758      | 0.000  | 2950         |
| (b) base + optimized prompt | 0.765  | 0.758      | 1.000  | 972          |
| (c) LoRA fine-tune          | 0.970  | 0.544      | 1.000  | 1284         |

> Lưu ý minh bạch: NB2 vô tình bị chạy 2 lần liên tiếp trong cùng notebook (cell trùng
> lặp khi gộp pipeline vào 1 lần "Save & Run All"). Hai lần chạy cho `target`/`format`
> giống hệt nhau, chỉ `latency` dao động nhỏ do nhiễu hệ thống (2956→2950ms, 966→972ms).
> Số trong bảng trên lấy từ lần chạy **cuối** vì đó là nội dung thực tế ghi vào
> `baselines_frozen.json` (file được đóng băng là file của lần ghi sau cùng).

**(b) có thật sự mạnh hơn (a) không?** Có — rõ rệt. (a) với prompt naive không sinh ra
được JSON hợp lệ (`target=0.000`, `format=0.000`): model trả lời tự do thay vì theo
schema 4 field yêu cầu, nên mọi câu đều bị chấm sai dù nội dung có thể đúng ý. (b) với
prompt tối ưu (ép schema + ví dụ định dạng) đạt `target=0.765`, `format=1.000`, đồng thời
nhanh hơn gần 3 lần (972ms so với 2950ms) vì không phải sinh phần giải thích dư thừa.
`regression` giống nhau (0.758) ở cả hai vì prompt chỉ thay đổi câu hỏi triage, không ảnh
hưởng tới 15 câu hỏi kiến thức chung. Không sửa `OPTIMIZED_PROMPT` sau khi đóng băng
(`optimized_prompt_sha=719e74d3b6232053`, dùng bản gốc trong repo). (c) fine-tune vượt cả
hai baseline ở `target` (0.970) và giữ `format=1.000`, nhưng đánh đổi bằng `regression`
tụt từ 0.758 xuống 0.544 — chi tiết ở §5.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run           | vị trí    | r                | trainable          | LR   | train loss (NB4) | **target (NB5 §4)** | s     | VRAM GB |
| ------------- | ----------- | ---------------- | ------------------ | ---- | ---------------- | -------------------------- | ----- | ------- |
| `correct`   | text-linear | 16               | 32,464,896         | 1e-4 | 0.6264           | **0.970**            | 827.6 | 8.78    |
| `attn_only` | q,v         | 283*(matched)* | 32,456,704         | 1e-4 | 0.5370           | **0.970**            | 723.8 | 8.79    |
| `wrong_lr`  | text-linear | 16               | 32,464,896         | 1e-5 | 1.5705           | **0.000**            | 848.9 | 8.78    |
| `qlora`     | text-linear | 16               | 32,464,896 (4-bit) | 1e-4 | 0.7058           | **0.940**            | 911.5 | 3.86    |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**
Trên tập target, `attn_only` **hoà** với `correct` — cả hai đều đạt `target=0.970` dù vị
trí gắn adapter hoàn toàn khác nhau (chỉ q,v so với toàn bộ 12 module tuyến tính). Thứ tự
đó KHÔNG giống thứ tự theo train loss: `attn_only` có `final_loss` thấp hơn (`0.537` so
với `0.6264` của `correct`), tức nếu xếp hạng bằng train loss thì `attn_only` sẽ "thắng",
nhưng trên chỉ số thật (target accuracy) nó chỉ hoà chứ không vượt. Điều này cho thấy ở
quy mô 250 mẫu và ngân sách tham số ~32.5M này, *rank* (được nâng lên 283 để bù cho việc
chỉ gắn vào q,v) đã đủ bù hoàn toàn cho *vị trí gắn adapter* hẹp hơn — rank cao bù được
vị trí, nhưng bù không có nghĩa là thắng; đây là bằng chứng mạnh rằng tổng dung lượng
tham số (param budget) quan trọng hơn việc đặt adapter ở "vị trí đúng" theo trực giác.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
`wrong_lr` chỉ giảm LR từ `1e-4` xuống `1e-5` (scale learning rate kiểu full-fine-tune
thay vì scale phù hợp cho LoRA), nhưng hậu quả rất lớn: loss dừng ở `1.5705` sau 30 bước,
cao gấp ~2.5 lần so với `correct` (`0.6264`), và đường loss giảm rất chậm, đều đặn (từ
2.163 → 1.12) thay vì rơi mạnh ở nửa sau epoch 1 như `correct`. Nếu chỉ nhìn con số loss
tuyệt đối mà không biết LR, ta dễ kết luận sai rằng model "đang học, chỉ chưa đủ bước" và
đề xuất train thêm — trong khi thực tế đây là LR quá thấp khiến optimizer gần như không
di chuyển: `target` trên tập eval của `wrong_lr` là `0.000`, tệ ngang với baseline (a)
chưa train, chứng tỏ model gần như không học được gì hữu ích dù loss có giảm chút ít.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` dùng peak VRAM `3.86GB` so với `8.78GB` của `correct` — tiết kiệm khoảng `56%`
(4.92GB). Cái giá phải trả: `final_loss` cao hơn (`0.7058` so với `0.6264`), thời gian
train lâu hơn (`911.5s` so với `827.6s`, do overhead quantize/dequantize), và quan trọng
nhất là `target` giảm xuống `0.940` so với `0.970` của `correct`/`attn_only` — mất 3 điểm
phần trăm độ chính xác. Số đo của mình **ủng hộ một phần** khuyến nghị "không dùng QLoRA
cho dòng model này": với tier T4 vốn đã đủ VRAM để chạy bf16/fp16 LoRA bình thường
(8.78GB < 14.6GB khả dụng), việc đánh đổi 3 điểm target để tiết kiệm VRAM là không đáng —
QLoRA chỉ hợp lý khi VRAM thực sự là nút thắt (ví dụ tier thấp hơn hoặc model lớn hơn).

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = −0.213` · `valid_trace_rate = 0.00`

Diễn giải: Cổng hồi quy FAIL vì `regression` tụt từ `0.758` (baseline b) xuống `0.544`
(fine-tune) — mức tụt `0.213` vượt xa ngưỡng cho phép `0.020` (tụt gấp hơn 10 lần ngưỡng).
Trong khi đó `target` lại cải thiện rất tốt, `+0.205` (từ 0.765 lên 0.970), nên nếu chỉ
nhìn mỗi target thì sẽ tưởng lầm đây là một fine-tune thành công. Nguyên nhân kỹ thuật
gần như chắc chắn là **catastrophic forgetting**: việc train 2 epoch chỉ trên 250 mẫu
JSON triage hẹp, không trộn thêm dữ liệu tổng quát (general instruction-following), đã
khiến model "quên" một phần khả năng trả lời 15 câu hỏi kiến thức chung trong tập
regression. `valid_trace_rate=0.00` chỉ phản ánh rằng dataset này không có reasoning
trace (`<think>` luôn rỗng), không liên quan tới nguyên nhân FAIL. Điều này nói lên rằng
đây **không phải** bài toán "model dở, cần train lâu hơn" mà là bài toán thiết kế dữ liệu:
cần trộn 1-5% dữ liệu tổng quát (deck §6.3) để giữ năng lực nền trong lúc vẫn học tốt tác
vụ hẹp. Một FAILED được đo đạc và lý giải rõ ràng như thế này có giá trị hơn nhiều so với
việc nới lỏng ngưỡng cho PASS giả tạo.

---

## 6. Định tính — bắt buộc có cả ca THUA

> Ghi chú: `qualitative.json` (NB5) chỉ log dự đoán của fine-tune (c), không log lại dự
> đoán (b) cho từng ticket cụ thể — cột "(b) prompt" dưới đây để trống vì không có số
> liệu per-example, không phải vì bỏ qua. Nhãn đúng lấy trực tiếp từ `data/eval_target.jsonl`.

| # | Ticket (rút gọn)                                            | Nhãn đúng                                                    | (b) prompt       | (c) fine-tune                                                                            | Nhận xét          |
| - | ------------------------------------------------------------- | --------------------------------------------------------------- | ---------------- | ---------------------------------------------------------------------------------------- | ------------------- |
| 1 | "...ốp lưng điện thoại...Shipper không gọi..."         | `van_chuyen/thap/ốp lưng điện thoại/tich_cuc`            | *(không log)* | khớp 4/4 (score 1.0)                                                                    | ✅ FT thắng        |
| 2 | "...ốp lưng điện thoại...Giá bao nhiêu..."             | `hoi_thong_tin/trung_binh/ốp lưng điện thoại/trung_tinh` | *(không log)* | khớp 4/4 (score 1.0)                                                                    | ✅ FT thắng        |
| 3 | "...ốp lưng điện thoại...Sai màu..."                    | `san_pham_loi/trung_binh/ốp lưng điện thoại/trung_tinh`  | *(không log)* | khớp 4/4 (score 1.0)                                                                    | ✅ FT thắng        |
| 4 | "...bình giữ nhiệt...Chưa thấy tiền. Khi nào tiện..." | `hoan_tien/thap/bình giữ nhiệt/tich_cuc`                   | *(không log)* | `urgency=trung_binh` (sai, đúng là `thap`); 3 field còn lại đúng (score 0.75) | ❌**FT thua** |
| 5 | "...nồi chiên không dầu...Thiếu phụ kiện..."           | `san_pham_loi/thap/nồi chiên không dầu/trung_tinh`        | *(không log)* | `urgency=trung_binh` (sai, đúng là `thap`); 3 field còn lại đúng (score 0.75) | ❌**FT thua** |
| 6 | "...áo khoác gió...Bị lỗi. Khi nào tiện..."            | `san_pham_loi/thap/áo khoác gió/tich_cuc`                  | *(không log)* | `urgency=trung_binh` (sai, đúng là `thap`); 3 field còn lại đúng (score 0.75) | ❌**FT thua** |

**Có mẫu chung nào ở các ca FT thua không?** Có — cả 3 ca thua đều sai **cùng một field**
(`urgency`), và sai theo **cùng một hướng**: model dự đoán `trung_binh` trong khi nhãn
đúng là `thap`. Cả 3 ticket đều chứa cụm từ lịch sự, không gấp gáp — "khi nào tiện", "cho
tôi hỏi", "cảm ơn shop nhiều" — nhưng model có vẻ học được quy tắc "có vấn đề về hàng/tiền
→ urgency ít nhất trung bình" từ phần lớn dữ liệu train, mà chưa phân biệt được tông giọng
lịch sự/không gấp của khách để hạ xuống `thap`. Đây là lỗi hệ thống (systematic), không
phải nhiễu ngẫu nhiên — gợi ý rằng nếu có thêm thời gian, nên bổ sung thêm ví dụ train có
urgency thấp đi kèm văn phong lịch sự để model học phân biệt tốt hơn.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Không, mình **chưa deploy** bản fine-tune `correct` này ở dạng hiện tại, dù
nó thắng áp đảo trên tập target (0.970 so với 0.765 của baseline b tốt nhất). Lý do là cổng
hồi quy FAILED với mức tụt `regression Δ=-0.213`, vượt xa ngưỡng cho phép 10 lần — nghĩa là
đổi lấy 20.5 điểm phần trăm độ chính xác triage, model đã đánh mất hơn 21 điểm phần trăm
khả năng trả lời đúng các câu hỏi kiến thức chung. Với một hệ thống thật, việc model "quên"
kiến thức nền trong lúc chạy production là rủi ro không chấp nhận được, kể cả khi task
chính (triage JSON) làm rất tốt. Đòn bẩy thật sự trong lab này, xếp theo mức ảnh hưởng đo
được: (1) **chất lượng/thành phần dữ liệu train** là đòn bẩy lớn nhất — chỉ 250 mẫu hẹp,
không trộn dữ liệu tổng quát, là nguyên nhân trực tiếp gây catastrophic forgetting, không
phải lỗi cấu hình LoRA; (2) **learning rate** là đòn bẩy nhị phân cực mạnh — `wrong_lr`
(chỉ đổi 1 con số) phá huỷ hoàn toàn khả năng học (target rơi về 0.000), cho thấy LR sai
có thể khiến toàn bộ công sức train vô nghĩa; (3) **vị trí gắn adapter** hoá ra là đòn bẩy
*yếu nhất* trong 3 thứ — `attn_only` với rank được bù (283) hoà hoàn toàn với `correct`
(text-linear, r=16) trên target, chứng tỏ tổng dung lượng tham số quan trọng hơn việc chọn
đúng vị trí theo trực giác; (4) **mask** đã đúng ngay từ đầu (`supervised_fraction=0.41`,
qua cả 2 assert) nên không phải là biến số trong case này, nhưng nếu sai thì mọi kết luận
ở trên đều vô nghĩa — đây là lý do rubric đặt nó ở vị trí nền tảng. Nếu làm lại, mình sẽ ưu
tiên sửa dữ liệu (trộn thêm 1-5% dữ liệu tổng quát) trước khi thử bất kỳ thay đổi cấu hình
LoRA nào khác.

**Ba điều tôi học được** (cụ thể, không generic):

1. Train loss thấp không đồng nghĩa với tốt hơn trên chỉ số thật: `attn_only` có
   `final_loss=0.537` thấp hơn `correct` (`0.6264`) nhưng hai run cho `target` HOÀ nhau
   (0.970 cả hai) — nếu chỉ nhìn loss để quyết định cấu hình nào "thắng" sẽ chọn nhầm.
2. Một fine-tune có thể thắng áp đảo ở đúng chỉ số bạn đang nhìn (target +0.205) và vẫn là
   một thất bại tổng thể nếu không đo thêm `regression` — nếu lab này không bắt buộc đo cả
   4 nhóm, mình chắc chắn đã kết luận sai là "thành công" chỉ vì nhìn mỗi target.
3. Một lỗi hệ thống (sai `urgency` ở câu lịch sự) chỉ lộ ra khi đọc từng ví dụ cụ thể
   (qualitative), không thấy được từ con số tổng (target=0.97 trông "gần như hoàn hảo") —
   số liệu tổng hợp che giấu loại lỗi lặp lại mà chỉ đọc ví dụ mới phát hiện được.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 10-15 mẫu dữ liệu tổng quát (general QA) vào
tập train theo tỉ lệ ~5% như deck §6.3 gợi ý, train lại `correct` với cùng 30 bước, rồi so
sánh `regression Δ` trước/sau để xác nhận giả thuyết catastrophic forgetting — nếu đúng,
đây sẽ là bằng chứng mạnh hơn hẳn so với suy luận hiện tại.

---

## Phụ lục — thưởng đã làm

- [X] B1 NB6 merge + hot-swap — trước merge `0.9700`, sau merge `0.9700` (Δ=0.0000, không
  tụt điểm), hot-swap thành công 3 adapter (`correct`, `attn_only`, `qlora`) trên cùng
  1 base, cả 3 cho cùng kết quả dự đoán trên ticket thử.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
