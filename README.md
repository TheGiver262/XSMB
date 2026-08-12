# XSMB predictive research

Dự án thu thập, kiểm định và dự báo Xổ số Miền Bắc cho hai không gian mục tiêu: **2 chữ số (00–99)** và **3 chữ số (000–999)**. Thiết kế ưu tiên dự báo ngoài mẫu, chống look-ahead và chống overfit thay vì biến các thống kê mô tả thành “cầu”.

## Kiến trúc dữ liệu

Dự án tách hai tầng:

- **Full history:** mirror dữ liệu chuẩn từ `khiemdoan/vietnam-lottery-xsmb-analysis`, hiện bắt đầu từ 2005. Dùng cho research, kiểm định thống kê, walk-forward và feature screening.
- **Rolling live:** `data/parts/` giữ **1.095 ngày lịch gần nhất (~3 năm)**. Đây là dữ liệu dùng để fit/calibrate model hằng ngày.

Full-history canonical mirror dưới `data/upstream/` gồm:

- `xsmb.csv`: kết quả đầy đủ từng giải;
- `xsmb-2-digits.csv`: đuôi 2 số của từng giải;
- `xsmb-sparse.csv`: số nháy 00–99 theo ngày;
- `metadata.json`: date range, số kỳ, checksum và kích thước file.

Nguồn upstream dùng MIT License; attribution và license được giữ trong `data/upstream/NOTICE.md` và `data/upstream/LICENSE_UPSTREAM`.

## Ba target dự báo

### 2 chữ số — `two_digit`

Một bộ 00–99 là hit nếu xuất hiện ở đuôi 2 số của ít nhất một trong 27 giải. Baseline fair-draw:

`1 - (99/100)^27 = 23.765729%`.

### 3 chữ số — `suffix3_any`

Một bộ 000–999 là hit nếu xuất hiện ở **3 số cuối** của bất kỳ giải nào có ít nhất 3 chữ số: ĐB, G1, G2, G3, G4, G5, G6. Tổng cộng 23 vị trí; G7 bị loại vì chỉ công bố 2 chữ số.

Baseline fair-draw:

`1 - (999/1000)^23 = 2.274876%`.

### 3 chữ số — `g6_exact`

Một bộ 000–999 là hit nếu trùng chính xác một trong ba kết quả G6.

Baseline fair-draw:

`1 - (999/1000)^3 = 0.2997001%`.

## Predictive research và feature gate

`src/research_predictive.py` chạy trên full history và chỉ sử dụng thông tin đứng trước kỳ cần đánh giá. Research hiện kiểm tra các nhóm signal như:

- long-run Bayesian rate;
- recent / rolling rate;
- weekday-conditioned rate;
- gap signal có giới hạn;
- thống kê từng số với z-score, p-value và Benjamini–Hochberg q-value để hạn chế multiple-testing false positives.

Signal muốn đi vào live model phải cải thiện **Brier score** tổng thể và thắng baseline ở ít nhất **60% các year-fold** trong walk-forward. Kết quả được ghi vào `research/feature_gate.json`.

Live model sau đó còn làm một vòng validation/calibration trên rolling 3 năm. Vì thế feature dù được phép vẫn có thể nhận `blend = 0`, tức forecast quay hoàn toàn về baseline nếu không tái lập được predictive edge.

Research đầu tiên trên full history cho thấy các recipe lịch sử thử nghiệm chưa thắng baseline một cách ổn định; feature gate hiện chỉ giữ `long_only` làm fallback tối thiểu và live calibration có quyền đưa tác động của nó về 0.

## Models

### 2 chữ số

`src/xsmb_probability_v2.py` dùng pooled ridge logistic regression và đồng thời chọn:

- feature set;
- ridge regularization;
- blend-to-baseline;

bằng expanding-window validation.

### 3 chữ số

`src/xsmb_3digit.py` dùng empirical-Bayes ensemble phù hợp hơn với không gian 1.000 outcome thưa. Recipe và blend được chọn walk-forward riêng cho `suffix3_any` và `g6_exact`.

## Pipeline live hằng ngày

Forecast và settlement được tách hoàn toàn:

1. **17:30 giờ Việt Nam:** tạo snapshot forecast trước giờ quay.
2. Forecast chỉ được nhìn dữ liệu đến ngày hôm trước và từ chối tạo hồi tố nếu kết quả target date đã tồn tại upstream.
3. Forecast live là immutable và có SHA-256 manifest.
4. **20:30**, fallback **21:30:** settlement lấy kết quả thực, chấm Brier/log-loss, cập nhật rolling 3 năm và retrain.
5. Theo dõi hiệu quả live trên cửa sổ 30/60/90 kỳ.

2-digit outputs:

- `forecasts/`
- `evaluation/daily_metrics.csv`
- `evaluation/rolling_summary.csv`
- `evaluation/model_runs.csv`

3-digit outputs:

- `forecasts_3d/suffix3_any/`
- `forecasts_3d/g6_exact/`
- `evaluation/3d_daily_metrics.csv`
- `evaluation/3d_rolling_summary.csv`
- `evaluation/3d_model_runs.csv`

## Workflows

- `daily-forecast.yml`: live forecast 2D + 3D.
- `daily-settle.yml`: settlement, rolling update, retrain và upstream mirror.
- `full-research.yml`: full-history research hằng tuần.
- `rebuild-rolling-3y.yml`: bootstrap/recovery rolling 1.095 ngày.
- `ci.yml`: pytest trên code changes.

Mọi workflow có quyền ghi repo dùng cùng một concurrency lock và sync latest `main` trước khi tạo output để tránh hai GitHub Actions cùng lúc ghi đè dataset/model artifacts.

## Chạy thủ công

```bash
python -m pip install -r requirements.txt
pytest -q

# Full-history mirror + research
python src/sync_upstream.py
python src/research_predictive.py

# Bootstrap/rebuild rolling 3 năm
python src/rebuild_rolling_3y.py

# Live 2-digit
python src/daily_pipeline_v2.py forecast
python src/daily_pipeline_v2.py settle

# Live 3-digit
python src/daily_3digit_pipeline.py forecast
python src/daily_3digit_pipeline.py settle
```

## Nguyên tắc diễn giải

Mục tiêu của hệ thống là kiểm tra xem lịch sử có tạo ra **predictive edge ngoài mẫu** hay không. Tần suất cao, chuỗi gan, weekday pattern hay một p-value nhỏ không tự động có giá trị dự báo. Khi evidence không đủ, output đúng của model là quay về fair-draw baseline chứ không ép phải chọn ra “số đẹp”.
