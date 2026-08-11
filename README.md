# XSMB 2-digit probability estimator

Dự án thống kê **2 năm** Xổ số Miền Bắc và ước lượng xác suất một bộ **00–99**
xuất hiện ít nhất một lần trong 27 kết quả của kỳ kế tiếp.

## Dữ liệu

- Cửa sổ: **2024-08-11 → 2026-08-10**
- Số ngày lịch: **730**
- Số kỳ XSMB có dữ liệu: **722**
- Năm trước: **361 kỳ** (2024-08-11 → 2025-08-10)
- Năm gần nhất: **361 kỳ** (2025-08-11 → 2026-08-10)
- Các ngày không có kỳ trong cửa sổ: 2025-01-28 → 2025-01-31 và 2026-02-16 → 2026-02-19
- Dataset trong repo: `data/parts/xsmb_part_01.csv` → `xsmb_part_08.csv`
- Metadata: `data/dataset_summary.json`
- Nguồn chính: `khiemdoan/vietnam-lottery-xsmb-analysis`
- Nguồn raw: https://raw.githubusercontent.com/khiemdoan/vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb.csv

## Thống kê mô tả

Các file sinh tự động:

- `output/statistics_2y_00_99.csv`: thống kê đầy đủ 00–99, gồm tần suất theo kỳ, tổng số nháy, so sánh hai năm, 30/90/180 kỳ, gan hiện tại, gan tối đa, chuỗi liên tiếp và giải đặc biệt.
- `output/statistics_2y_summary.csv`: các chỉ số tổng quan và cực trị nổi bật.

## Định nghĩa xác suất

Một bộ 2 số được tính là “xuất hiện” nếu hai chữ số cuối của nó xuất hiện ở **ít nhất
một trong 27 giải** của một kỳ XSMB.

Nếu mỗi đuôi 2 số là độc lập và đều trên 00–99, xác suất lý thuyết cho một bộ bất kỳ là:

`p0 = 1 - (99/100)^27 = 23.765729%`

Đây là baseline quan trọng. Lịch sử không tự động làm một số “đến lượt” phải xuất hiện.

## Thuật toán

Thuật toán dùng một mô hình **ridge logistic pooled** cho 100 bộ số, với các feature chỉ
được tính từ dữ liệu quá khứ:

- tỷ lệ xuất hiện 7 / 30 / 90 kỳ gần nhất;
- tỷ lệ xuất hiện theo 27 vị trí trong 30 kỳ;
- tỷ lệ dài hạn có Bayesian shrinkage về baseline;
- tỷ lệ theo thứ trong tuần có Bayesian shrinkage;
- số kỳ kể từ lần xuất hiện gần nhất;
- có/không xuất hiện ở kỳ trước.

Để hạn chế overfit:

1. dùng warm-up 90 kỳ;
2. chọn regularization bằng expanding-window time-series validation;
3. chọn hệ số blend `lambda` bằng Brier score;
4. xác suất cuối = `p0 + lambda * (p_model - p0)`.

### Kết quả khi dùng 722 kỳ

Backtest chọn:

- `selected_l2 = 10`
- `selected_blend = 0.00`
- `CV Brier = 0.1810720336`
- `Baseline Brier = 0.1810720336`
- `Brier improvement = 0`

Nghĩa là với bộ dữ liệu 2 năm này, các feature lịch sử **không cải thiện dự báo ngoài mẫu** so với baseline. Vì vậy mô hình được thiết kế để tự quay về xác suất lý thuyết **23.765729% cho mỗi bộ 00–99**, thay vì ép nhiễu lịch sử thành tín hiệu dự báo.

## Chạy

```bash
python -m pip install -r requirements.txt
python src/extend_history.py
python src/build_statistics.py
pytest -q
python src/xsmb_probability.py \
  --data data/parts \
  --target-date 2026-08-11 \
  --out output/prediction_2026-08-11.csv
```

## Lưu ý

Kết quả là **ước lượng thống kê có kiểm định backtest**, không phải cam kết dự đoán xổ số.
Với một quy trình quay công bằng, mọi bộ 00–99 có xác suất cơ bản như nhau và phần lớn
dao động lịch sử chỉ là nhiễu ngẫu nhiên.
