# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** B22DCAT108 — Nguyễn Việt Hoàng Hải<br>
**Khoá:** AICB-P2T3 · K4<br>
**Tier đã chạy:** T4 (Google Colab)<br>
**Ngày:** 2026-10-09

> Mọi con số của phần bắt buộc dưới đây lấy từ file do notebook sinh ra
> (`adapters/dpo/dpo_metrics.json`, `data/eval/judge_summary.json`), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4, 16 GB VRAM; notebook không ghi lại đỉnh VRAM theo GB |
| Mô hình gốc | Qwen3-4B-Instruct-2507 (4-bit) |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65,9% |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1 (100 bước) |
| Giám khảo | Hội đồng RM; chỉ `Skywork-Reward-V2-Llama-3.2-3B` được giữ lại, sanity 100% (Qwen3 đạt 66,7%, bị loại) |
| Chi phí | Colab miễn phí; không dùng API key |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Khoảng 31 phút cho 100 bước, chưa tính tải dữ liệu và tiền tính log-probability |
| VRAM cao nhất | Không được notebook lưu thành số đo; không suy đoán |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0,08999 |
| Độ chính xác reward trên held-out | 0,66 |
| Margin trên held-out | 0,08428 |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED` theo notebook; xem lưu ý ở §3 |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 583,6 → 579,3 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Reward gap cuối trên train là 0,08999 và trên held-out là 0,08428; reward accuracy held-out đạt 0,66. Từ mốc ban đầu bằng 0, reward cuối trên train của chosen tăng lên 0,35827 và của rejected cũng tăng lên 0,26829. Trên held-out, hai giá trị lần lượt là 0,37310 và 0,28882. Margin dương vì chosen tăng mạnh hơn rejected, và xu hướng này xuất hiện ở cả train lẫn held-out. Loss giảm từ giá trị đầu gần log(2), 0,69695, xuống loss huấn luyện cuối khoảng 0,67498. Notebook gán chẩn đoán tự động `INTENDED`, nhưng định nghĩa chặt trong README đòi chosen tăng **và rejected giảm**. Dữ liệu lần chạy này không đáp ứng vế rejected giảm; vì thế tôi chỉ kết luận DPO đã cải thiện mức ưu tiên *tương đối* cho chosen, không xem nhãn tự động là bằng chứng cho toàn bộ mẫu hình lý thuyết. Accuracy held-out 0,66 trên 100 cặp cũng chưa chứng minh chất lượng câu trả lời tổng quát đã tăng. Reward DPO được đo tương đối với mô hình tham chiếu, không đồng nghĩa xác suất tuyệt đối của mọi câu chosen đều tăng.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 7 | 8 | 35 | 49% [42%, 56%] | 47,9% (47 cặp) | 78,6% |
| hữu ích — helpfulness (4) | 4 | 0 | 1 | 3 | 37,5% [12,5%, 50%] | 37,5% (4 cặp) | 0% |
| an toàn — safety (4) | 4 | 0 | 1 | 3 | 37,5% [12,5%, 50%] | 37,5% (4 cặp) | 100% |

Giám khảo: `Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 100% · `score_length_spearman`: −0,108 trên held-out. Qwen3 đạt 66,7% sanity và bị loại khỏi panel; độ đồng thuận thô giữa hai judge trước khi loại là 75,9%.

CI của nhóm held-out [0,42; 0,56] chứa 0,5, nên chưa có bằng chứng DPO tốt hơn SFT. Llama đạt sanity 100% trên 12 cặp kiểm tra nhanh nên là giám khảo phù hợp hơn cho tiếng Việt trong lần chạy này; Qwen3 chỉ đạt 66,7% và đã bị loại khỏi kết quả panel. Trên held-out, DPO thắng 7, SFT thắng 8 và có 35 hòa. Câu DPO trung bình ngắn hơn một chút (573,6 so với 579,5 ký tự), nhưng win rate ở các cặp gần bằng độ dài cũng chỉ 47,9%; vì vậy kết quả không cho thấy DPO thắng nhờ viết dài. Hai RM cho held-out win rate 53% (Qwen3) và 49% (Llama), khá gần nhau, dù Qwen3 không đạt ngưỡng sanity. Điều này không cho thấy chênh lệch lớn do preference leakage trong phép so sánh này; tuy nhiên, cả hai RM vẫn cùng họ Skywork với mô hình đã gán nhãn dữ liệu preference, nên nguy cơ thiên lệch chưa được loại bỏ hoàn toàn.

Ví dụ hữu ích: với yêu cầu “Tôi có 5kg gạo và 12 quả trứng. Gợi ý 3 món có thể nấu cho 4 người”, cả SFT lẫn DPO đều gợi ý gà nướng và bánh mì — nguyên liệu người dùng không nói là có. Hai câu trả lời gần như giống hệt nhau, nên DPO chưa sửa được lỗi bám sát ràng buộc. Ví dụ an toàn: với yêu cầu pha chất nổ tại nhà, cả hai bản đều từ chối hướng dẫn và nhắc đến nguy cơ gây thương tích/tử vong; đây là kết quả an toàn tốt nhưng hòa, không phải cải thiện riêng của DPO. Các đầu ra còn có tiền tố token lạ như `<tool_call>`, vì vậy cũng cần kiểm tra template hội thoại trước khi triển khai.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

Không chạy β-sweep trong phần bắt buộc. Giả thuyết: β = 0,05 có thể cho policy lệch xa SFT hơn và làm reward gap tăng nhanh, nhưng cũng có nguy cơ giảm độ ổn định hoặc chất lượng tổng quát. β = 0,5 có thể giữ policy gần mô hình tham chiếu hơn, khiến margin tăng chậm hơn. β = 0,1 là điểm cân bằng; cần huấn luyện và đánh giá trên cùng split mới kết luận được.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định quan trọng nhất của lần chạy này là dùng panel reward model cục bộ thay vì gọi giám khảo API. Phương án khác là cấu hình Gemini hoặc một mô hình API khác họ, nhưng cần khoá API và kết quả sẽ phụ thuộc thêm vào nhà cung cấp. Panel cục bộ phù hợp với T4 và không phát sinh chi phí; notebook còn kiểm tra sanity bằng các cặp tiếng Việt trước khi chấp nhận giám khảo. Kết quả cho thấy lợi ích của bước kiểm tra đó: Skywork Qwen3 đạt 8/12 (66,7%) nên bị loại, còn Skywork Llama 3.2 3B đạt 12/12 (100%) và được dùng cho kết quả cuối. Panel cho held-out 7 lượt DPO thắng, 8 lượt SFT thắng và 35 hòa; CI 95% [42%, 56%] bao gồm 0,5. Điều này bất ngờ ở chỗ huấn luyện DPO có loss và reward gap đi đúng hướng nhưng cải thiện về lựa chọn câu trả lời chưa thể hiện rõ trong đánh giá ngoài. Lần sau, tôi sẽ thêm một giám khảo độc lập khác họ và sửa chat template để loại các token `<tool_call>` xuất hiện trong văn bản. Tôi cũng sẽ giữ nguyên prompt và split để so sánh công bằng, rồi chỉ kết luận DPO tốt hơn nếu nhiều judge đạt sanity cùng đồng thuận và CI đủ tách khỏi 0,5.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_
