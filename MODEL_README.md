# XSMB Model Methodology, Validation & V4 Assessment

> Tài liệu kỹ thuật mô tả các family model đang/đã được nghiên cứu trong repo XSMB, công thức chính, quy trình validation, giới hạn diễn giải và lý do `stable75_no_digit` (V4) hiện là challenger tốt nhất theo bằng chứng development. **“Tốt nhất” ở đây không đồng nghĩa đã chứng minh có lợi thế dự đoán chắc chắn hoặc lợi nhuận thực tế.**

## 1. Bài toán và baseline lý thuyết

### 1.1 Lô tô 2 chữ số

Mỗi kỳ XSMB có 27 vị trí giải. Với một số 2 chữ số `n ∈ {00,...,99}`, biến mục tiêu presence là:

```text
y[t,n] = 1 nếu n xuất hiện ít nhất một lần trong 27 đuôi 2 số của kỳ t
         0 nếu không xuất hiện
```

Nếu coi 27 đuôi 2 số là IID đều trên 100 giá trị, xác suất một số xuất hiện ít nhất một lần trong ngày là:

```text
P0 = 1 - (99/100)^27
   = 0.23765728565289646
   ≈ 23.765729%
```

Đây là baseline chính cho mọi model presence 2D.

Số lần xuất hiện kỳ vọng của một số trong 27 vị trí là `27 × 0.01 = 0.27` nháy/kỳ.

### 1.2 Cặp hai số

Với hai suffix khác nhau `a,b`:

```text
P(any hit)  = 1 - (98/100)^27
            ≈ 42.043247%
```

Xác suất cả hai số cùng xuất hiện ít nhất một lần trong một kỳ là:

```text
P(both hit) = 1 - 2(99/100)^27 + (98/100)^27
            ≈ 5.488210%
```

Không được lấy `P0 × P0` để tính `both-hit`, vì tại một vị trí giải duy nhất không thể đồng thời mang hai suffix khác nhau. Công thức trên là inclusion-exclusion đúng cho 27 vị trí.

Số nháy kỳ vọng của hai số cộng lại là `0.54` nháy/kỳ.

### 1.3 Hai mục tiêu 3 chữ số

Repo tách 3D thành hai bài toán khác nhau:

- `suffix3_any`: 000–999 xuất hiện ở đuôi 3 số trong các vị trí đủ 3 chữ số từ ĐB đến G6, tổng 23 vị trí đủ điều kiện.
- `g6_exact`: 000–999 xuất hiện chính xác trong ba kết quả G6.

Baseline lý thuyết:

```text
P0_suffix3 = 1 - (999/1000)^23 ≈ 2.274876%
P0_g6      = 1 - (999/1000)^3  ≈ 0.2997001%
```

Hai target này phải được train/score riêng. Không được dùng ranking 2D rồi ghép tùy tiện thành 3D và gọi đó là xác suất model G6.

---

## 2. Metric và kiểm định

### 2.1 Brier score

Với `M` quan sát nhị phân:

```text
BS = (1/M) Σ_i (y_i - p_i)^2
```

Brier càng thấp càng tốt. Khi so với fair baseline:

```text
Brier improvement = BS_baseline - BS_model
```

Giá trị dương mới là tốt hơn baseline.

### 2.2 Log loss

```text
LL = -(1/M) Σ_i [ y_i ln(p_i) + (1-y_i) ln(1-p_i) ]
```

Log loss càng thấp càng tốt và phạt mạnh dự báo quá tự tin nhưng sai.

### 2.3 Top-k hit rate

Với mỗi kỳ chọn `k` số có score cao nhất:

```text
HitRate_k = Σ_t Σ_{n∈TopK(t)} y[t,n] / (k × T)
```

Đây là hit rate **trên mỗi pick**, không phải xác suất “ít nhất một số trong bộ k trúng”.

Lift tính theo điểm phần trăm:

```text
Lift_pp = 100 × (HitRate_k - P0)
```

### 2.4 Z-score nhị thức đơn giản

```text
z = (p_hat - P0) / sqrt(P0(1-P0)/N)
```

Z-score này chỉ là diagnostic. Sau model selection, temporal dependence và thử nhiều candidate, z naïve có thể lạc quan. Vì vậy V4 còn dùng nested validation, selection-adjusted test và resampling.

### 2.5 Multiple testing: Benjamini-Hochberg

Khi test nhiều số/feature/pair, p-value thô dễ sinh false positive. Với `m` p-value đã sắp xếp `p_(1) ≤ ... ≤ p_(m)`, BH kiểm soát False Discovery Rate bằng ngưỡng dạng:

```text
p_(i) ≤ i × q / m
```

V4.0 dùng logic multiple-testing này trong audit tín hiệu trước khi xây model.

### 2.6 Resampling dùng trong V4

- **Circular-shift null**: dịch vòng chuỗi outcome so với signal để phá alignment dự đoán nhưng giữ cấu trúc thời gian nội tại.
- **Block bootstrap**: lấy lại mẫu theo block nhiều ngày để giữ local autocorrelation tốt hơn bootstrap từng ngày độc lập.
- **Paired sign-flip**: đổi dấu chênh lệch model-vs-baseline theo fold để kiểm tra liệu lợi thế có ổn định giữa các fold hay không.
- **LOYO (Leave-One-Year-Out)**: bỏ từng năm một để xem kết luận có phụ thuộc vào riêng một năm hay không.

---

## 3. Nguyên tắc chống look-ahead và provenance

Đây là phần quan trọng ngang với bản thân thuật toán.

1. Forecast cho ngày `D` chỉ được dùng outcome đến trước `D`.
2. File forecast phải được tạo trước khi kết quả `D` tồn tại.
3. Forecast đã tạo là **write-once/immutable**; settlement chỉ chấm, không sửa lại dự báo.
4. Mọi output phải có `target_date` và `data_cutoff`.
5. Chỉ được dùng file khi `target_date` đúng chính xác ngày cần dự báo.
6. Không được lấy output ngày trước rồi gọi là forecast ngày hiện tại chỉ vì ranking trông hợp lý.
7. V4 frozen còn kiểm model signature và môi trường deterministic.
8. Post-cutoff outcomes có thể dùng làm **lagged state features** trước target, nhưng không được refit/tune/select lại frozen model.
9. Nếu upstream trễ, manual state chỉ được merge khi có provenance rõ ràng và conflict detection; không được hạ freshness guard.

Quy tắc số 4–6 đặc biệt quan trọng: một output `target_date=2026-08-11`, `data_cutoff=2026-08-10` không được dùng làm forecast cho 12/08.

---

## 4. Fair baseline

Fair baseline không cố ranking số nào hơn số nào:

```text
p[n] = P0, ∀ n
```

Vai trò của baseline:

- xác định mức mà model phải vượt qua;
- kiểm tra calibration;
- cho phép model fallback khi tín hiệu không đủ mạnh;
- ngăn việc “bắt buộc phải dự đoán edge” trong dữ liệu vốn rất gần ngẫu nhiên.

Một model an toàn phải có quyền trả về baseline. Trong các model v2/v3, điều này được thực hiện bằng `blend = 0`.

---

## 5. Live 2D logistic model v2

### 5.1 Kiến trúc

Live 2D dùng pooled logistic/ridge framework trên 100 suffix, với cửa sổ rolling khoảng 1.095 ngày và feature-set được giới hạn bởi `research/feature_gate.json`.

Mô hình logistic tổng quát:

```text
z[t,n] = β0 + β1 x1[t,n] + ... + βp xp[t,n]
p_raw[t,n] = sigmoid(z[t,n]) = 1 / (1 + exp(-z[t,n]))
```

Regularization L2 hạn chế coefficient lớn do noise:

```text
loss = logloss + λ ||β||²
```

Trong code, strength được tìm qua các candidate tương đương ridge/L2; không nên quy đổi trực tiếp mọi tham số implementation sang một `λ` toán học duy nhất nếu solver/scaling khác nhau.

### 5.2 Nhóm feature

Các feature-set được nghiên cứu theo mức phức tạp tăng dần:

- `long_only`: chỉ tín hiệu tần suất dài hạn.
- `recency`: thêm các rate ngắn/trung hạn.
- `recency_gap`: thêm gap/recency.
- `full`: toàn bộ feature được định nghĩa trong base feature extractor, bao gồm các nhóm thời gian/vị trí được gate cho phép.

Mục tiêu của feature gate là không cho một feature vừa nhìn tốt in-sample tự động đi vào live model.

### 5.3 Hyperparameter và calibration

Validation tìm L2 trong một grid gồm các mức như:

```text
10, 100, 300, 1000, 3000
```

Sau khi có `p_raw`, model shrink về fair baseline:

```text
p_final = P0 + b × (p_raw - P0),  0 ≤ b ≤ 1
```

`b` được chọn theo validation. Nếu `b=0`:

```text
p_final = P0
```

nghĩa là model chủ động kết luận “không đủ edge để khác fair baseline”.

### 5.4 Validation

Selection dùng các fold thời gian chronological, không random shuffle. Tiêu chí chính là Brier rồi log loss, với feature-set bị giới hạn bởi full-history feature gate.

**Đánh giá:** đây là live architecture tốt về governance, nhưng khi gate/final blend quay về 0 thì không nên diễn giải ranking phụ như một predictive edge đã được duyệt.

---

## 6. Empirical-Bayes 3D: `suffix3_any` và `g6_exact`

Hai target 3D dùng mô hình empirical Bayes thủ công để tránh overfit trên không gian 1.000 class rất sparse.

### 6.1 Long-term posterior

Với `h` lần hit trong `N` kỳ, prior strength `s` và fair probability `p0`:

```text
p_long = (h + s p0) / (N + s)
```

Đây là shrinkage: sample ít thì estimate nằm gần `p0`; sample lớn mới được phép rời baseline đáng kể.

### 6.2 Recent windows

Cùng công thức trên nhưng `h,N` chỉ lấy trong cửa sổ recent:

```text
p_short = (h_short + s_short p0) / (N_short + s_short)
p_recent_long = (h_long + s_long p0) / (N_long + s_long)
```

`g6_exact` dùng window/prior dài hơn `suffix3_any` vì baseline chỉ khoảng 0.3%, dữ liệu mỗi số rất sparse.

### 6.3 Weekday posterior

```text
p_weekday = (h_weekday + s_weekday p0) / (N_weekday + s_weekday)
```

Chỉ dùng các kỳ quá khứ cùng thứ trong tuần.

### 6.4 Gap signal

Gap được cap để tránh một số quá gan tạo extrapolation vô hạn. Tín hiệu gap chỉ được phép điều chỉnh rất nhỏ quanh baseline, sau đó còn qua recipe weight và final blend.

### 6.5 Recipe

Các recipe hiện tại gồm:

```text
long_only
recency
weekday
gap
```

Ví dụ dạng tổng quát:

```text
p_recipe = w1 p_long + w2 p_short + w3 p_recent_long
         + w4 p_weekday + w5 p_gap
```

Sau đó:

```text
p_final = p0 + blend × (p_recipe - p0)
```

Nếu validation chọn `blend=0`, toàn bộ 000–999 trở về cùng fair probability.

**Kết luận hiện tại:** riêng `g6_exact`, các lần kiểm gần đây thường chọn `blend=0`; do đó chưa có cơ sở model để nói một đuôi 3 số G6 cụ thể tốt hơn các đuôi khác.

---

## 7. V3 multi-horizon empirical-Bayes probability challenger

Source hiện hành: `src/xsmb_multihorizon_v3.py`.

### 7.1 Horizon

Model tạo empirical-Bayes estimate trên nhiều thang:

```text
90 ngày
365 ngày
1095 ngày (~3 năm)
1825 ngày (~5 năm)
3650 ngày (~10 năm)
full history
```

Component cơ bản:

```text
p_component = (hits + s P0) / (exposure + s)
```

với `s` là prior strength, ví dụ 100, 300 hoặc 1000 tùy recipe.

### 7.2 Exponential decay

Với half-life `H`:

```text
λ = exp(log(0.5) / H)
num_t = λ num_(t-1) + y_t
den_t = λ den_(t-1) + 1
```

Sau đó weighted hits/exposure được đưa vào cùng empirical-Bayes shrinkage.

Half-life chính: 365, 1095 và 1825 ngày.

### 7.3 Recipe tiêu biểu

Các recipe đơn:

```text
w1y_s100, w1y_s300
w3y_s100, w3y_s300, w3y_s1000
w5y_s100, w5y_s300, w5y_s1000
w10y_s100, w10y_s300
full_s100, full_s300
decay1y_s300, decay3y_s300, decay5y_s300
```

Các recipe hỗn hợp:

```text
multi_3_5_full_s300
  = 0.45 × w3y + 0.35 × w5y + 0.20 × full

multi_1_3_5_10_full_s300
  = 0.10 × w1y + 0.30 × w3y + 0.25 × w5y
  + 0.20 × w10y + 0.15 × full

decay_mix_s300
  = 0.20 × d1y + 0.50 × d3y + 0.30 × d5y

hybrid_s300
  = 0.25 × w3y + 0.20 × w5y + 0.15 × w10y + 0.10 × full
  + 0.10 × d1y + 0.20 × d3y
```

### 7.4 Nested selection

Mỗi outer test year chỉ được chọn recipe/blend bằng bốn năm trước đó.

Blend grid:

```text
0.00, 0.05, 0.10, ..., 1.00
```

Final:

```text
p = P0 + blend × (p_raw - P0)
```

Promotion probability yêu cầu đồng thời:

- Brier improvement > 0;
- log-loss improvement > 0;
- ít nhất 60% outer years có Brier improvement dương;
- mean top-5 không thấp hơn baseline.

### 7.5 Kết quả

V3 probability challenger **FAIL promotion**:

- nested Brier hơi tệ hơn fair baseline;
- nested log loss hơi tệ hơn fair baseline;
- chỉ 3/9 outer years có Brier improvement dương.

Điều này cho thấy việc thấy một recipe hiện tại như `w1y_s300` xếp một số lên đầu không đủ để gọi đó là model tốt nhất toàn hệ thống.

---

## 8. V3 ranking challenger

Source: `src/xsmb_rank_challenger.py`.

Mục tiêu của model này khác probability model: ưu tiên **thứ hạng top-k**, không cố khẳng định score là xác suất calibrated.

### 8.1 Tie-aware percentile rank

Mỗi recipe probability được chuyển sang percentile rank. Nếu nhiều số bằng điểm, chúng nhận average rank, tránh bias do số thứ tự `00 < 01 < ...`.

### 8.2 Consensus rankers

Ngoài các horizon ranker đơn, model tạo:

```text
consensus_1_3_5    = mean(rank_1y, rank_3y, rank_5y)
consensus_3_5_10   = mean(rank_3y, rank_5y, rank_10y)
consensus_all      = mean(rank_1y, rank_3y, rank_5y, rank_10y, rank_full)
consensus_decay    = mean(rank_decay1y, rank_decay3y, rank_decay5y)
stable_hot_1_3_5  = min(rank_1y, rank_3y, rank_5y)
```

`stable_hot_1_3_5` đòi một số phải tương đối cao ở cả ba horizon, thay vì chỉ bùng ở một horizon.

### 8.3 Bayesian shrinkage cho performance của ranker

Với hit count `h`, trials `N`, prior strength 1000:

```text
p_post = (h + 1000 P0) / (N + 1000)
```

### 8.4 Objective chọn ranker

```text
core = 0.20 p_top1 + 0.55 p_top3 + 0.25 p_top5
objective = core
          - 0.50 × std(top3_rate_by_year)
          + 0.002 × (positive_year_fraction - 0.5)
```

Nếu `positive_year_fraction < 0.5`, có thêm penalty `0.005`.

Như vậy model ưu tiên top-3 nhưng phạt ranker chỉ thắng nhờ một vài năm đặc biệt.

### 8.5 Kết quả pooled OOS

```text
Top-1: 23.6552%  | lift -0.1105 pp | z -0.144
Top-3: 24.1845%  | lift +0.4188 pp | z  0.947 | 7/9 outer years dương
Top-5: 24.1996%  | lift +0.4339 pp | z  1.266
```

Promotion gate yêu cầu top-3 `z ≥ 1.96`, vì vậy **FAIL**.

Đây là tín hiệu ranking đáng theo dõi, nhưng chưa đủ mạnh để thay V4 frozen hoặc để coi score percentile là xác suất trúng.

---

## 9. Legacy unified: presence, intensity và direct-combo

Các family này là lineage nghiên cứu trước V4. Một phần source lịch sử không còn nằm trong active `main`; vì vậy phần này chỉ ghi các công thức/khái niệm đã được xác nhận trong các run và không tự dựng chi tiết implementation đã mất.

### 9.1 Presence

Presence chỉ quan tâm:

```text
số n có xuất hiện ít nhất một lần trong kỳ hay không
```

Family này ổn định hơn intensity vì target đúng với nhu cầu “có về hay không” và ít bị nhiễu bởi một kỳ ra 2–4 nháy.

Pooled OOS top-3 từng đạt khoảng:

```text
24.1845% vs fair 23.7657%
lift ≈ +0.4188 pp
z ≈ 0.95
7/9 năm dương
```

Có hướng tích cực nhưng chưa significant.

### 9.2 Intensity

Intensity cố mô hình hóa số nháy/27 vị trí thay vì chỉ presence. Trong unified weighting sau đánh giá, presence nhận trọng số 1 còn intensity 0. Điều đó nghĩa là intensity không bổ sung đủ predictive value sau OOS validation.

### 9.3 Direct pair/combo

Direct-combo dự đoán event hai số cùng xuất hiện. Fair both-hit phải so với `5.488210%`.

Kết quả pooled OOS:

```text
both-hit ≈ 5.5736%
fair     ≈ 5.4882%
lift     ≈ +0.0853 pp
z        ≈ 0.21
4/9 năm dương
```

Một pair từng có training estimate khoảng `7.32%`, nhưng đó **không phải forward probability đã chứng minh**. OOS cho thấy edge combo rất yếu.

---

## 10. Sequential adaptive vs frozen

Một nhánh nghiên cứu test cập nhật model theo từng ngày theo kiểu prequential.

Adaptive first-pass trên 361 draws:

```text
top1      26.0388%  (+2.273 pp, z≈1.01)
pair-any  44.8753%  (+2.832 pp, z≈1.09)
pair-both  5.5402%  (+0.052 pp, z≈0.04)
```

Tuy nhiên Brier/logloss kém fair baseline và adaptive thua phiên bản frozen cùng initial state:

```text
frozen pair-any  46.5374%
frozen pair-both  6.0942%
mean occurrences  0.60388
```

Kết luận nghiên cứu: **slow/frozen adaptation đáng tin hơn cập nhật nhanh theo lỗi gần nhất**. Đây là một lý do V4 cố tình freeze selection/model thay vì retune mỗi ngày.

---

## 11. Mean reversion / gap research

Giả thuyết thường gặp là “số lâu không ra thì sắp ra”. Nếu draw IID, điều đó không đúng:

```text
P(hit ngày mai | đã miss k ngày) = P0
```

Research mean-reversion/gap không tìm được edge đủ credible. Vì vậy:

- gap có thể là một feature nhỏ để model kiểm tra interaction;
- không được dùng logic “gan lâu nên xác suất cao” như quy tắc quyết định độc lập;
- tương tự, một số chạy 6 ngày liên tiếp không tự động có xác suất ngày thứ 7 cao hoặc thấp hơn nếu không có OOS evidence.

---

# 12. V4: từ audit đến `stable75_no_digit`

V4 khác các đời trước không chỉ ở classifier mà chủ yếu ở **research protocol**.

## 12.1 V4.0 — Null audit

Development data khoảng 7.157 kỳ / 193.239 positions được audit trước khi xây model mạnh hơn.

Kết quả chính:

- overall chi-square Monte Carlo `p ≈ 0.182`;
- không có marginal/pair/temporal signal nào sống sót FDR đủ thuyết phục.

Kết luận: dữ liệu nhìn tổng thể rất gần fair null. Vì vậy V4 không xuất phát từ giả định “chắc chắn có quy luật”, mà từ câu hỏi hẹp hơn: **có ranking signal nhỏ, ổn định, có thể sống qua OOS hay không?**

## 12.2 V4.1 — Bayesian Presence

Bayesian probability model không thắng calibration metrics; fallback về uniform/fair là hợp lý. Legacy ranker top-3 chỉ khoảng 24.0856%, lift +0.3199 pp, z≈0.68. Gate FAIL.

## 12.3 V4.2 — Ranking experts

Nhiều ranking experts được thử. Candidate tốt nhất vẫn có CI đi qua 0; gate FAIL. Bài học: không promote chỉ vì average lift dương.

## 12.4 V4.3 — Tabular logistic

V4 chuyển sang supervised tabular classifier với feature đa horizon và vị trí giải.

Bản đầu `logit_no_digit` cho top-3 khoảng 24.6361%, nhưng audit phát hiện ablation chưa sạch vì một số interaction còn phụ thuộc nhóm bị loại. Kết quả đó không được coi là xác nhận cuối.

## 12.5 V4.3.1 — Clean nested ablation

Sau khi làm sạch dependency:

```text
Top-3 ≈ 24.6361%
Diff vs legacy ≈ +0.5505 pp
Positive folds: 5/8
Paired CI95: [-0.7706, +1.8593] pp
Selection-adjusted p ≈ 0.3949
```

FAIL. Dù average tốt hơn, uncertainty còn quá lớn.

## 12.6 V4.3.2 — `stable75_no_digit`

Feature được giữ khi có consistency đủ cao qua các fold; threshold frozen là `0.75`. Nhóm `digit` và `digit_joint` bị loại.

Nested OOS 2021–2025:

```text
Top-1: 23.99278%
Top-3: 25.31569% | z≈2.5720
Top-5: 25.07517%

Legacy top-3 trong cùng protocol: 23.79234%
Legacy top-5 trong cùng protocol: 23.81239%

V4 - legacy top-3: +1.523351 pp
V4 - legacy top-5: +1.262778 pp
Positive years: 3/5
Paired CI95 top-3 diff: [-0.080176, +3.027160] pp
Selection-adjusted p (best-of-3): 0.0769693
```

Theo preregistered promotion gate, **vẫn FAIL**, vì CI còn chạm qua 0 và selection-adjusted p chưa dưới 0.05.

Year-by-year top-3 diff vs legacy:

```text
2021: -0.6463 pp
2022: +2.6778 pp
2023: +3.1394 pp
2024: -1.1971 pp
2025: +5.0459 pp
```

Không thể bỏ hai năm âm rồi chỉ báo ba năm dương.

## 12.7 V4.3.3 — Frozen robustness

Sau khi cố định rule, V4 dùng thêm các test không còn chọn lại rule:

```text
Circular-shift observed diff: +1.523351 pp
95th percentile null:         +1.342954 pp
one-sided p:                   0.03307   -> PASS riêng test này
```

Block bootstrap CI:

```text
7d:  [-0.040, +3.087] pp
14d: [-0.060, +3.087] pp
28d: [-0.060, +3.007] pp
56d: [+0.0195,+3.0267] pp
```

Các test khác:

```text
weighted 5-fold sign-flip p = 0.125  -> FAIL
LOYO minimum diff ≈ +0.9919 pp       -> PASS
all-block-CI-positive               -> FALSE
overall confirmation                -> FALSE
```

Không được cherry-pick `p=0.03307` rồi tuyên bố V4 đã chứng minh edge. Tổng bộ robustness vẫn chưa xác nhận hoàn toàn.

---

## 13. V4 frozen model specification

### 13.1 Frozen classifier

`stable75_no_digit` dùng logistic regression regularized:

```text
z[n] = β0 + Σ_j β_j x[n,j]
score[n] = sigmoid(z[n])
```

Frozen hyperparameter:

```text
C = 0.1
```

Đây là L2-regularized sklearn logistic setting. Không nên đồng nhất máy móc `C=0.1` với một `λ=10` trong mọi cách scale loss.

### 13.2 13 selected features

Authoritative frozen feature list:

```text
1.  hit_rate_30
2.  hit_rate_90
3.  position_rate_30
4.  hit_rate_365
5.  hit_rate_1095
6.  hit_rate_1825
7.  bayes_long_rate
8.  position_db_90
9.  position_g3g5_90
10. position_g6_90
11. position_g7_90
12. accel_30_90
13. gap_x_recent
```

Ý nghĩa:

- `hit_rate_W`: tỷ lệ các kỳ trong W kỳ trước mà suffix xuất hiện ít nhất một lần.
- `position_rate_30`: tần suất nháy trên toàn bộ 27 vị trí trong recent 30, khác presence vì một ngày có thể nhiều nháy.
- `bayes_long_rate`: long-run presence được shrink về `P0`, theo dạng empirical-Bayes `(h+sP0)/(N+s)`.
- `position_db_90`: rate ở vị trí ĐB trong 90 kỳ gần nhất.
- `position_g3g5_90`: rate trên nhóm vị trí G3–G5 trong 90 kỳ.
- `position_g6_90`: rate ở G6 trong 90 kỳ.
- `position_g7_90`: rate ở G7 trong 90 kỳ.
- `accel_30_90`: feature mô tả mức thay đổi short-horizon so với medium-horizon.
- `gap_x_recent`: interaction giữa trạng thái gap và hoạt động recent.

**Lưu ý reproducibility:** source branch V4 research cũ hiện không còn là active branch trên `main`, nên exact transform archival của `accel_30_90`, `gap_x_recent` và exact prior strength của `bayes_long_rate` không được suy đoán lại trong tài liệu này. Khi source archival được restore, README nên cập nhật exact expression trực tiếp từ code.

### 13.3 Vì sao loại digit features?

`stable75_no_digit` loại nhóm `digit` và `digit_joint` vì các pattern hàng chục/hàng đơn vị dễ tạo apparent signal khi thử rất nhiều combination. Việc loại chúng sau clean ablation giảm dimensionality và giảm nguy cơ data-mining artifact.

### 13.4 Frozen training boundary

```text
fit window start:       2022-01-01
training outcome cutoff:2025-08-10
consistency threshold:  0.75
C:                      0.1
```

Model signature:

```text
8beb784a75ab8c9fd641a70b4a5ceb783e26d08f29edfec0433b154757d56f4e
```

Frozen environment:

```text
Python        3.12.13
NumPy         2.5.2
SciPy         1.18.0
scikit-learn  1.9.0
```

BLAS/thread environment được khóa để giảm non-determinism.

### 13.5 Score không phải calibrated probability

Dù logistic có output trong `[0,1]`, V4 frozen đang dùng output chủ yếu để **xếp hạng**. Không được đọc `score=0.251` thành “xác suất trúng 25.1%” nếu calibration chưa được chứng minh.

Điều được phép nói:

```text
52 có ranking score cao hơn 54
```

Điều không được tự động nói:

```text
52 có xác suất trúng chính xác 25.1%
```

---

# 14. Tại sao V4 hiện là model tốt nhất?

Cách gọi chính xác là:

> **V4 `stable75_no_digit` là challenger mạnh nhất hiện tại theo development evidence và chất lượng research protocol; chưa phải predictive edge đã được xác nhận prospectively.**

Có sáu lý do chính.

### 14.1 So sánh trực tiếp cùng protocol tốt hơn legacy

Trong nested 2021–2025 cùng comparator:

```text
Top-3 V4    25.31569%
Top-3 legacy23.79234%
Diff        +1.523351 pp

Top-5 V4    25.07517%
Top-5 legacy23.81239%
Diff        +1.262778 pp
```

Đây là comparison mạnh hơn việc lấy hai con số từ hai giai đoạn khác nhau rồi so trực tiếp.

### 14.2 Feature đa horizon nhưng không quá rộng

V4 kết hợp:

- short-term 30/90;
- medium 365;
- long 1095/1825;
- position-specific rates;
- Bayesian long rate;
- chỉ hai interaction chính.

Sau stable selection còn 13 feature, đủ linh hoạt hơn frequency-only nhưng nhỏ hơn một feature zoo dễ overfit.

### 14.3 Stable feature selection

Threshold consistency `0.75` khiến feature phải sống qua nhiều fold mới được giữ. Đây là khác biệt quan trọng so với chọn feature chỉ dựa trên full-sample importance.

### 14.4 Regularization mạnh

`C=0.1` hạn chế coefficient cực đoan. Với XSMB, true edge nếu tồn tại có khả năng nhỏ; regularization mạnh phù hợp hơn model cố fit mọi fluctuation.

### 14.5 Validation nghiêm hơn các đời trước

V4 không dừng ở một hit-rate đẹp. Nó dùng:

- nested OOS;
- paired CI;
- selection-adjusted p-value;
- circular-shift null;
- block bootstrap;
- sign-flip;
- LOYO;
- frozen rule trước prospective test.

Do đó “tốt nhất” đến từ cả performance lẫn chất lượng bằng chứng.

### 14.6 Frozen prospective protocol

Model đã được khóa thay vì sửa sau mỗi kết quả. Đây là điều cần thiết để cuối cùng biết edge có thật hay chỉ là research overfit.

---

## 15. Vì sao vẫn chưa được gọi là “đã chứng minh tốt”?

Các bằng chứng chống lại kết luận quá mạnh:

```text
positive nested years = 3/5
paired CI95 vẫn đi qua 0
selection-adjusted p = 0.07697 > 0.05
weighted sign-flip p = 0.125
all-block-CI-positive = false
overall robustness confirmation = false
```

Ngoài ra prospective OOS thực sự chỉ bắt đầu từ 12/08/2026. Development/backtest dù sạch đến đâu vẫn không thay thế được chuỗi forecast bất biến trong tương lai.

V4 mạnh nhất ở **top-3/top-5 ranking**, không phải top-1 certainty. Nested top-1 chỉ 23.99278%, rất gần fair 23.76573%.

---

## 16. Prospective OOS protocol của V4

Prospective start: `2026-08-12`.

Các invariants:

1. forecast phải tồn tại trước kết quả target;
2. file write-once;
3. training outcomes không vượt `2025-08-10`;
4. kết quả sau cutoff chỉ được đi vào lagged state feature trước target;
5. không retune coefficient, feature selection, hyperparameter;
6. latest state phải đúng `target_date - 1 day`;
7. target result chưa được tồn tại lúc forecast;
8. cùng ngày sau 18:00 ICT không tạo forecast mới;
9. model signature phải khớp frozen manifest;
10. pair operational chỉ là top-2 ranking, **không phải calibrated pair-probability model**;
11. không backfill forecast sau khi đã biết kết quả.

Forecast prospective hợp lệ đầu tiên cho 12/08/2026:

```text
Top-1: 52
Top-3: 52, 54, 13
Top-5: 52, 54, 13, 85, 80
Operational top-2 pair: 52, 54
```

Đây là snapshot để chấm về sau, không phải lý do để chỉnh model nếu một ngày đầu tiên miss hoặc hit.

---

## 17. Bảng đánh giá tổng hợp

> Các dòng khác evaluation window không được dùng để tính một ranking tuyệt đối bằng phép trừ trực tiếp. Bảng dùng để mô tả strength/weakness; comparison V4-vs-legacy đáng tin nhất là comparison trong cùng nested protocol V4.

| Model/family | Kỹ thuật chính | Bằng chứng tốt nhất | Vấn đề chính | Trạng thái |
|---|---|---|---|---|
| Fair baseline | IID analytical | P0=23.7657% | Không ranking | Baseline |
| Live 2D v2 | pooled logistic + L2 + feature gate + blend | Có anti-lookahead/fallback | Gate thường buộc về baseline | Conservative live |
| 3D EB | Bayesian shrinkage + recency/weekday/gap + blend | Stable với sparse 3D | `g6_exact` thường blend=0 | Chưa có edge |
| V3 probability | multi-horizon EB + exponential decay + nested selection | nhiều horizon, calibration | Brier/logloss pooled không thắng; 3/9 positive | FAIL |
| V3 ranking | percentile consensus + Bayesian performance shrinkage | top3 24.1845%, 7/9 positive | z≈0.95, gate fail | Promising weak |
| Legacy presence | frequency/presence ranking | pooled top3 ~24.18% | lift nhỏ, z~0.95 | Weak evidence |
| Direct combo | direct pair both-hit | ~5.5736% vs 5.4882% | z~0.21 | FAIL |
| Adaptive sequential | prequential updates | vài metric ngắn hạn cao | Brier/logloss worse; thua frozen | Không ưu tiên |
| Mean reversion | gap/absence hypothesis | không có credible edge | gambler’s-fallacy risk | Không dùng độc lập |
| **V4 stable75** | stable features + regularized logistic + nested/robustness + freeze | **top3 25.3157%; +1.523 pp vs same-protocol legacy** | CI/p adjusted/sign-flip chưa pass toàn bộ | **Best challenger, not confirmed** |

---

## 18. Promotion policy đề xuất

Không promote V4 sang “confirmed best predictive model” chỉ vì vài ngày prospective đầu tiên tốt.

Nên yêu cầu tối thiểu:

- đủ số lượng prospective draws định trước;
- đánh giá top-1/top-3/top-5 đúng metric đã preregister;
- compare với fair baseline và frozen legacy comparator;
- không đổi rule giữa sample;
- CI paired không còn đi qua 0 cho metric chính;
- robustness giữa month/quarter/year hợp lý;
- Brier/logloss chỉ được dùng như probability evidence nếu output được calibration; ranking score không được ngụy trang thành probability;
- mọi model selection mới phải diễn ra ở generation kế tiếp, không sửa V4 retrospective.

Nếu V4 thất bại prospective, kết luận đúng là “development edge không replicate”, không phải tiếp tục tune cho đến khi thắng.

---

## 19. Reproducibility và source lineage

Các thành phần live/v3 hiện nằm trên `main`, gồm các source như:

```text
src/xsmb_multihorizon_v3.py
src/xsmb_rank_challenger.py
src/research_predictive.py
research/feature_gate.json
paper/challenger/
```

V4 được phát triển trên research lineage riêng và sau đó freeze bằng manifest/signature. Research branch cũ hiện không còn là active ref truy cập được từ current `main`. Vì vậy:

- các con số validation/frozen spec trong tài liệu này là record của V4 research protocol;
- không tự chế lại exact archived transform nếu source chưa restore;
- nếu cần tái huấn luyện V4 từ đầu, bước đầu tiên phải là restore exact source/manifest và verify model signature trước khi sửa bất kỳ thứ gì.

---

# 20. Kết luận

XSMB 2D có fair presence baseline khoảng **23.7657% mỗi số mỗi kỳ**. Phần lớn các family đã thử chỉ tạo lift nhỏ và không vượt qua đầy đủ OOS/statistical gate. Điều quan trọng nhất từ research đến nay không phải tìm được một “công thức xổ số chắc thắng”, mà là loại dần các signal dễ overfit.

`stable75_no_digit` V4 hiện đứng đầu vì:

1. thắng legacy rõ nhất trên cùng nested development protocol ở top-3/top-5;
2. feature selection ổn định và loại digit artifacts;
3. dùng regularized logistic trên tập feature vừa phải;
4. được kiểm bằng nhiều robustness test hơn các đời trước;
5. model/spec/environment đã freeze;
6. đã chuyển sang prospective write-once OOS thay vì tiếp tục tối ưu backtest.

Nhưng kết luận thống kê cuối cùng vẫn là:

> **V4 là model tốt nhất để tiếp tục kiểm nghiệm prospectively, chưa phải model đã chứng minh tạo predictive edge bền vững hoặc lợi nhuận thực tế.**

Mọi forecast hằng ngày phải giữ ranh giới rõ giữa **rank score**, **calibrated probability**, **backtest evidence** và **prospective evidence**.