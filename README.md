# XSMB predictive research

Dự án thu thập, kiểm định và dự báo Xổ số Miền Bắc theo hướng **out-of-sample, chống look-ahead và chống overfit**. `main` là branch vận hành và là source of truth duy nhất cho code mới.

> **Trạng thái 2026-09-10:** lineage nghiên cứu `top3-walkforward` V1 → V2 → V3 → V4 → **V5 set10** đã được hợp nhất vào `main` qua PR #11. V5 là research model mới nhất ở mức **set**, không thay thế kết luận thống kê của official live model và chưa được coi là predictive edge đã xác nhận.

## Bài toán đang theo dõi

### Official 2D — `two_digit`

Một suffix 00–99 là hit nếu xuất hiện ít nhất một lần trong 27 đuôi 2 số của kỳ XSMB.

Fair baseline:

```text
P(hit) = 1 - (99/100)^27 ≈ 23.765729%
```

Live model: `src/xsmb_probability_v2.py`, pooled ridge logistic regression + validation + shrink/blend về fair baseline. Nếu validation chọn `blend=0`, model chủ động kết luận không có preference đủ tin cậy và toàn bộ calibrated probability trở về baseline.

### Official 3D

- `suffix3_any`: 000–999 xuất hiện ở 3 số cuối của 23 vị trí đủ 3 chữ số. Fair baseline ≈ **2.274876%**.
- `g6_exact`: 000–999 trùng một trong ba kết quả G6. Fair baseline ≈ **0.2997001%**.

Code: `src/xsmb_3digit.py`. Hai target được train/score riêng bằng empirical-Bayes; không ghép ranking 2D thành 3D.

### V5 set10 — latest research

V5 đổi mục tiêu vận hành từ ranking top-3 sang **chọn đúng 10 suffix khác nhau**, với strict success:

```text
WIN iff có >= 3 suffix khác nhau trong bộ 10 xuất hiện trong 27 vị trí XSMB
```

Random-set analytical baseline của điều kiện này là **43.89569%**. Spec đầy đủ: `V5_SPEC.md`. Code chính:

- `src/walkforward_set10_v5.py`
- `src/v5_daily_forecast.py`
- `src/walkforward_top3_v2.py` — frozen feature/ranking dependency được V5 tái sử dụng.

V5.0 hiện chọn scheme **`marginal_top10`**. Pair interaction và regime/reverse interaction đều **không qua ablation gate** (chỉ 2/6 fold dương), nên `pair_lambda=0` và `reverse_lambda=0`.

Development selection (6 fold × 365 draw) của `marginal_top10`:

```text
median strict >=3/10 : 47.1233%
mean strict >=3/10   : 46.5297%
worst fold           : 42.4658%
random baseline      : 43.8957%
```

Không được đọc các số trên như xác nhận độc lập. Burned benchmark từ 2025-08-11 đến 2026-09-09 chỉ đạt **40.1535%**, thấp hơn random baseline khoảng **3.74 điểm %**. Khoảng này đã tồn tại trước khi V5 được thiết kế nên chỉ dùng để báo cáo, không dùng làm confirmation set.

Prospective ledger hiện có **5 forecast hợp lệ**, 3 WIN / 2 LOSS = **60%**, nhưng mẫu quá nhỏ; checkpoint đầu tiên được đặt ở **7 forecast hợp lệ**. Các ngày 2026-09-02 đến 2026-09-04 đang được đánh dấu missing/invalid và không được backfill.

Artifacts:

- `forecasts/v5_set10/YYYY-MM-DD.json` — immutable dated snapshots
- `forecasts/set10_v5_next.json` — current generated candidate
- `evaluation/v5_set10_prospective/` — prospective settlement ledger
- `evaluation/walkforward_set10_v5_*` — development/burned diagnostics

`src/v5_daily_forecast.py` có hard deadline **18:00 Asia/Ho_Chi_Minh** và từ chối tạo retrospective snapshot nếu chưa có snapshot hợp lệ trước deadline. Từ khi V5 được merge, workflow `v5-set10-daily-forecast.yml` chạy trực tiếp từ **`main`** và ghi snapshot trở lại `main`; không còn phụ thuộc branch research V5 để vận hành.

## Các model/challenger khác

- **V3 probability/ranking challenger:** vẫn được refresh để nghiên cứu multi-horizon; ranking score không phải calibrated probability.
- **V4 `stable75_no_digit`:** archival/frozen development evidence được mô tả trong `MODEL_README.md`. Không tự dựng forecast V4 mới nếu source + frozen prospective manifest không tồn tại trên `main`.
- **V1–V4 top3 walk-forward:** là lineage lịch sử dẫn tới V5; không phải branch vận hành riêng sau khi V5 đã được merge.

## Dữ liệu

Dự án tách hai tầng:

- **Full history:** mirror từ `khiemdoan/vietnam-lottery-xsmb-analysis`, bắt đầu từ 2005; dùng cho research/walk-forward.
- **Rolling live:** `data/parts/`, khoảng **1.095 ngày (~3 năm)**; dùng cho fit/calibration live.

Canonical mirror dưới `data/upstream/`:

- `xsmb.csv`
- `xsmb-2-digits.csv`
- `xsmb-sparse.csv`
- `metadata.json`

Nguồn upstream dùng MIT License; attribution nằm trong `data/upstream/NOTICE.md` và `data/upstream/LICENSE_UPSTREAM`.

## Workflow trên `main`

- **09:00 VN, fallback 12:00 — `V5 set10 pre-draw forecast`** (`v5-set10-daily-forecast.yml`): chạy V5 từ `main`, tạo/kiểm tra immutable set10 snapshot; fallback là no-op nếu snapshot primary đã tồn tại. Hard lock 18:00.
- **15:00 VN — `Multi-horizon v3 challenger`** (`multihorizon-v3.yml`): refresh V3 probability/ranking research. Temporary immutable paper week đã kết thúc sau 2026-08-17; workflow hiện không tạo paper snapshot mới sau mốc đó.
- **16:00 VN — `Daily pre-draw forecast`** (`daily-forecast.yml`): tạo immutable official 2D + 3D snapshots. Khi đánh giá prospective validity phải dùng thời điểm run/generation thực tế, không chỉ cron khai báo.
- **20:30 VN, fallback 21:30 — `Daily post-draw settlement`** (`daily-settle.yml`): settle 2D/3D, cập nhật rolling data, retrain và mirror upstream.
- `full-research.yml`: full-history research định kỳ.
- `rebuild-rolling-3y.yml`: bootstrap/recovery rolling window.
- `ci.yml`: pytest trên code changes.

Các workflow ghi `main` dùng chung concurrency lock `xsmb-repo-write` để hạn chế race khi nhiều Actions cùng cập nhật dữ liệu/model artifacts.

## Chạy thủ công

```bash
python -m pip install -r requirements.txt
pytest -q

# Official research/live
python src/sync_upstream.py
python src/research_predictive.py
python src/rebuild_rolling_3y.py
python src/daily_pipeline_v2.py forecast
python src/daily_pipeline_v2.py settle
python src/daily_3digit_pipeline.py forecast
python src/daily_3digit_pipeline.py settle

# V5 set10 research
python src/walkforward_set10_v5.py

# V5 immutable daily forecast; chỉ chạy pre-draw
python src/v5_daily_forecast.py
```

## Branch policy

- `main` là authoritative branch cho vận hành và phát triển tiếp theo.
- Các branch `analysis/*`, `temp/*`, retrospective theo ngày và các branch V1–V4 riêng lẻ là lineage/experiment cũ; không dùng chúng làm nguồn forecast hiện tại.
- V5 đã được merge vào `main`; workflow V5 trên `main` cũng đã được chuyển sang checkout/push `main` thay vì branch research.
- Sau khi một research lineage được merge vào `main`, mọi thay đổi tiếp theo nên bắt đầu từ `main` thay vì tiếp tục phát triển trên branch cũ.

## Nguyên tắc diễn giải

Không có model nào được gọi là “có edge” chỉ vì một vài ngày trúng hoặc một backtest trung bình dương. Forecast cho ngày D chỉ được dùng thông tin trước D; snapshot prospective phải được khóa trước giờ quay; không backfill; không retune theo một ngày; không cherry-pick subset tốt.

Nếu validation/calibration quay về fair baseline thì đó là output hợp lệ của hệ thống. V5 hiện là **latest set-level research**, nhưng burned benchmark đang dưới baseline và prospective sample mới có 5 ngày hợp lệ, vì vậy **chưa có đủ bằng chứng để khẳng định predictive edge hoặc lợi nhuận thực tế**.

Chi tiết methodology và historical V4 assessment: `MODEL_README.md`.