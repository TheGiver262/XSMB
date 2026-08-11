# XSMB 2-digit probability estimator

Dự án thống kê **2 năm** Xổ số Miền Bắc và ước lượng xác suất một bộ **00–99** xuất hiện ít nhất một lần trong 27 kết quả của kỳ kế tiếp.

## Dữ liệu

- Seed window ban đầu: **2024-08-11 → 2026-08-10**
- Số ngày lịch: **730**
- Số kỳ XSMB ban đầu: **722**
- Dataset được duy trì dưới `data/parts/` và tự dịch theo **730 ngày lịch gần nhất** sau mỗi kỳ quay mới.
- Metadata hiện hành: `data/dataset_summary.json`
- Nguồn chính: `khiemdoan/vietnam-lottery-xsmb-analysis`
- Nguồn raw: https://raw.githubusercontent.com/khiemdoan/vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb.csv

## Thống kê mô tả

Các file sinh tự động:

- `output/statistics_2y_00_99.csv`: thống kê đầy đủ 00–99, gồm tần suất theo kỳ, tổng số nháy, so sánh hai năm, 30/90/180 kỳ, gan hiện tại, gan tối đa, chuỗi liên tiếp và giải đặc biệt.
- `output/statistics_2y_summary.csv`: các chỉ số tổng quan và cực trị nổi bật.

## Định nghĩa xác suất

Một bộ 2 số được tính là “xuất hiện” nếu hai chữ số cuối của nó xuất hiện ở **ít nhất một trong 27 giải** của một kỳ XSMB.

Nếu mỗi đuôi 2 số là độc lập và đều trên 00–99, xác suất lý thuyết cho một bộ bất kỳ là:

`p0 = 1 - (99/100)^27 = 23.765729%`

Đây là baseline quan trọng. Lịch sử không tự động làm một số “đến lượt” phải xuất hiện.

## Thuật toán

Thuật toán dùng một mô hình **ridge logistic pooled** cho 100 bộ số, với các feature chỉ được tính từ dữ liệu quá khứ:

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

### Kết quả seed với 722 kỳ

Backtest chọn:

- `selected_l2 = 10`
- `selected_blend = 0.00`
- `CV Brier = 0.1810720336`
- `Baseline Brier = 0.1810720336`
- `Brier improvement = 0`

Nghĩa là với seed 2 năm hiện tại, các feature lịch sử **không cải thiện dự báo ngoài mẫu** so với baseline. Mô hình được thiết kế để tự quay về xác suất lý thuyết nếu tín hiệu không qua được backtest.

## Pipeline live hằng ngày

Pipeline tách forecast và settlement để tránh look-ahead:

1. **17:30 giờ Việt Nam**: `.github/workflows/daily-forecast.yml` tạo forecast cho ngày hiện tại trước giờ quay. Dữ liệu bị khóa ở ngày hôm trước. Forecast lưu dưới `forecasts/YYYY-MM-DD.csv` và **không bao giờ bị ghi đè**.
2. `forecasts/manifest.csv` lưu thời điểm tạo, ngày cutoff và SHA-256 của từng forecast/metrics file để kiểm chứng snapshot.
3. **20:30 giờ Việt Nam**, với fallback lúc **21:30**: `.github/workflows/daily-settle.yml` kiểm tra kết quả thực tế. Nếu hôm đó không có kỳ hoặc nguồn chưa cập nhật thì pipeline không ghi settlement giả.
4. Khi có kết quả, forecast live của đúng ngày được chấm bằng **Brier score** và **log-loss** so với baseline lý thuyết.
5. Dataset được dịch thành rolling 730 ngày, thống kê được làm mới, model được retrain/backtest và tạo `output/next_preview.csv` cho ngày kế tiếp.
6. `evaluation/daily_metrics.csv` lưu từng kết quả live; `evaluation/rolling_summary.csv` tổng hợp cửa sổ **30 / 60 / 90 kỳ đã chấm**; `evaluation/model_runs.csv` lưu lịch sử hyperparameter và blend.

Chốt chống hồi tố: `python src/daily_pipeline.py forecast` sẽ từ chối tạo forecast nếu kết quả của ngày mục tiêu đã tồn tại trong nguồn upstream.

## Chạy thủ công

```bash
python -m pip install -r requirements.txt
pytest -q

# Tạo forecast live cho hôm nay (theo múi giờ Việt Nam)
python src/daily_pipeline.py forecast

# Settlement sau khi kết quả hôm nay đã xuất hiện ở nguồn
python src/daily_pipeline.py settle
```

Các script seed/rebuild vẫn giữ lại:

```bash
python src/extend_history.py
python src/build_statistics.py
python src/xsmb_probability.py \
  --data data/parts \
  --target-date 2026-08-11 \
  --out output/prediction_2026-08-11.csv
```

## Lưu ý

Kết quả là **ước lượng thống kê có kiểm định backtest**, không phải cam kết dự đoán xổ số. Với một quy trình quay công bằng, mọi bộ 00–99 có xác suất cơ bản như nhau và phần lớn dao động lịch sử có thể chỉ là nhiễu ngẫu nhiên.
