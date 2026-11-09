# Ghi chép kết quả chạy: tiền xử lý, chọn tham số/ngưỡng và kết quả mô hình

Đồ án SECOM – Ứng dụng AI trong công nghiệp. File này tổng hợp **số liệu thật** từ output đã lưu của `02_preprocessing.ipynb`, `03_modeling.ipynb`, `04_evaluation.ipynb`, `artifacts/decision_config.json` và `artifacts/results/*.csv`. Không có số nào được ước lượng; số nào không có nguồn thì ghi "không có".

**Thiết lập chung:** `SEED = 42`, `TEST_SIZE = 0.20`, `MISSING_THRESHOLD = 0.70`; nhãn 0 = Pass, 1 = Fail; chia stratified 80/20 **trước** mọi phép fit; timestamp chỉ để mô tả, không đưa vào X. Hai mô hình: **Logistic Regression (LR)** và **Random Forest (RF)**.

**Cập nhật (ngưỡng cho LR):** sau lần xem test đầu tiên, nhóm **thêm** một ngưỡng chọn từ OOF-train cho LR (mục 3.6 và 4.1, 4.4) để so sánh LR với RF cùng điều kiện. Ngưỡng chỉ chọn từ OOF-train, không dùng test; **model cuối vẫn là RF @ 0,14 và verdict Improve không đổi**. Các số cũ của RF và của LR @ 0,5 giữ nguyên.

| Tập | Số dòng | Pass | Fail | Tỷ lệ Fail |
|---|---|---|---|---|
| Toàn bộ | 1.567 | 1.463 | 104 | 6,64% |
| Train | 1.253 | 1.170 | 83 | 6,62% |
| Test | 314 | 293 | 21 | 6,69% |

(Nguồn: 02 mục 2; 03 và 04 kiểm tra lại cùng các số này bằng `get_split`, không lưu chỉ số chia ra file.)

---

## 1. Tiền xử lý (notebook 02)

Pipeline 4 bước, **fit chỉ trên train**, test chỉ `transform`:
`MissingRateSelector(0.70) → SimpleImputer(median) → VarianceThreshold(0) → StandardScaler`.

### 1.1. Kết quả từng bước

| Bước | Tham số | Học từ | Kết quả trên train (1.253 dòng) | Kết quả trên test (314 dòng) |
|---|---|---|---|---|
| Dữ liệu thô | – | – | 590 cột | 590 cột |
| Lọc cột thiếu > 70% | ngưỡng 0,70 (quy ước của lab, không phải chuẩn công nghiệp) | tỷ lệ thiếu từng cột của train | loại **8** cột → 582 cột | 582 cột |
| Điền thiếu | trung vị | trung vị từng cột của train | 582 cột, không còn NaN | 582 cột, không còn NaN |
| Bỏ cột hằng | phương sai = 0 | phương sai của train sau khi điền | loại **116** cột → 466 cột | 466 cột |
| Chuẩn hoá | `StandardScaler` | μ và σ của train | 466 cột; \|mean\| lớn nhất ≈ 3,9e-14, std = 1 | 466 cột; \|mean\| lớn nhất 0,183, std từ 0 đến 2,29 |

Số cột thiếu > 70% (8) và cột hằng (116) khi học trên train **trùng đúng** số đếm trên toàn bộ dữ liệu ở notebook 01 (8 và 116; 590 → 582 → 466).

### 1.2. Lý do từng bước (dẫn từ EDA, notebook 01)

| Vấn đề | Bằng chứng ở notebook 01 | Quyết định |
|---|---|---|
| Cột thiếu quá nhiều | 8 cột thiếu > 70% (4 cột 91,19%, 4 cột 85,58%); ngưỡng 70% hay 80% cho cùng kết quả | Loại, ngưỡng học trên train |
| Thiếu rải rác | 41.951 ô thiếu (4,54%); mỗi mẫu thiếu 4–152 ô, không mẫu nào đầy đủ | Điền trung vị, không xoá dòng |
| Cột hằng | 116 cột phương sai 0; 6 cột gần hằng | Loại cột hằng; cột gần hằng để F-test quyết định |
| Thang đo | std chênh khoảng 6,3e7 lần | `StandardScaler`, dùng chung cho LR và RF |
| Outlier | 110/464 cột > 5% ngoài biên IQR; 269/474 cột \|skew\| > 2 | **Giữ nguyên**, không xoá, không cắt |
| Mất cân bằng | Fail 6,64% | Giữ nhãn; so sánh class weight ở notebook 03 |
| Số chiều lớn | 466 cột so với 83 Fail; 329 cặp \|r\| > 0,95 | Chọn đặc trưng bằng F-test trong pipeline ở notebook 03 |

### 1.3. Kiểm tra sau tiền xử lý

- Không còn NaN hay inf ở train và test; số dòng và nhãn giữ nguyên; train và test cùng 466 cột.
- **Kiểm tra rò rỉ:** `imputer.statistics_` trùng trung vị của train; trung vị train và test khác nhau ở **423/582** cột (nếu imputer học từ test thì kết quả điền sẽ khác). `scaler.mean_` và `scale_` trùng thống kê của train.
- Từng bước so với đầu ra pipeline: sai khác lớn nhất **0,0** (train và test).
- **Outlier còn lại sau scale:** 1.804 trên 583.898 giá trị train (0,31%) có \|z\| > 5, rải ở 345/466 cột, lớn nhất \|z\| = 35,38.
- Lưu `artifacts/preprocessing_pipeline.joblib` (khoảng 26,6 KB), nạp lại và transform: sai khác 0,0, `np.allclose` đều True.

Lưu ý: test có cột với std = 0 sau khi điền thiếu (mọi giá trị test của cột đó bằng nhau); đây không phải lỗi.

---

## 2. Chọn đặc trưng (notebook 03, mục 3)

- `SelectKBest(f_classif, k = 40)` nằm **bên trong** pipeline đầy đủ nên được fit lại trong từng fold của CV; k = 40 cố định, không tuning.
- Để xem kết quả, tiền xử lý và F-test được fit một lần trên **toàn bộ train** (chỉ minh hoạ, không đưa vào CV), rồi ánh xạ về **chỉ số cột gốc 0–589** (qua các mask `missing_rate → constant → SelectKBest`; độ dài các bước khớp: 582 → 466 → 40). Ánh xạ làm **một lần** ở 03; notebook 04 đọc lại từ `decision_config.json`.

**15 cột có F-score cao nhất (cột gốc):** 103 (38,35), 510 (26,41), 59 (20,15), 348 (15,72), 129 (14,66), 431 (14,09), 64 (13,68), 21 (12,46), 125 (11,31), 430 (10,88), 434 (10,76), 100 (10,65), 316 (10,19), 435 (10,00), 436 (9,58).

**40 cột gốc được chọn:** 14, 21, 28, 59, 64, 65, 70, 95, 100, 103, 114, 121, 122, 124, 125, 127, 129, 130, 160, 163, 180, 210, 213, 247, 249, 295, 298, 316, 348, 351, 365, 387, 430, 431, 434, 435, 436, 452, 455, 510.

- Welch T-test trên train (minh hoạ phụ, không dùng để chọn): top 40 theo \|t\| trùng **22/40** cột với F-test.
- Khi chạy T-test, scipy báo `RuntimeWarning` về mất độ chính xác khi dữ liệu gần giống nhau (một số cột gần như không đổi trong một nhóm); giữ nguyên cảnh báo, không xử lý thêm vì T-test chỉ để minh hoạ.

---

## 3. Cách chọn tham số, mô hình và ngưỡng (notebook 03)

### 3.1. Khung đánh giá

- **Pipeline đầy đủ nhận dữ liệu thô:** `MissingRateSelector → median imputer → VarianceThreshold → StandardScaler → SelectKBest(k=40) → Classifier`. Trong CV, mỗi fold `clone` rồi fit lại cả chuỗi từ dữ liệu thô của fold (không lấy ma trận đã biến đổi ở notebook 02 đưa vào CV).
- **CV:** `StratifiedKFold(5, shuffle=True, random_state=42)` chỉ trên train; một bộ fold dùng chung cho mọi so sánh. Mỗi fold: fit 1.002–1.003 mẫu, validation 250–251 mẫu (16–17 Fail).
- Ngưỡng ở bước so sánh CV là **0,5** (mặc định của `predict`). Mean ± std là giá trị trung bình và độ lệch chuẩn (ddof = 0) giữa 5 fold; BER mỗi fold = 1 − Balanced Accuracy.
- **Không dùng test** để chọn bất cứ thứ gì trong notebook 03.

### 3.2. Tham số của hai mô hình

| | Logistic Regression | Random Forest |
|---|---|---|
| Cấu hình | `C=1.0`, `max_iter=2000`, solver `lbfgs`, `random_state=42` | `n_estimators=200`, `max_features="sqrt"`, `max_depth=None`, `min_samples_leaf=1`, `random_state=42`, `n_jobs=-1` |
| `class_weight` | `"balanced"` (chọn bằng CV, xem 3.3) | `"balanced"` (cấu hình cơ sở của thầy; không so sánh các mức) |
| k của F-test | 40 (cố định) | 40 (cố định) |
| Cách chọn các tham số còn lại | Cấu hình cơ sở của thầy, **không tuning** (không GridSearch) | Cấu hình cơ sở của thầy, **không tuning** |
| Hội tụ | 60 vòng lặp trên giới hạn 2.000, không có `ConvergenceWarning` | không áp dụng |
| Ngưỡng chốt riêng | **0,836**, chọn từ OOF-train bằng cùng quy tắc F1, **thêm sau lần xem test đầu tiên** chỉ để so sánh cùng điều kiện (xem 3.6); ngoài ra dùng 0,5 | **0,14** (vì là model cuối, xem 3.5) |

### 3.3. Chọn class weight cho LR (CV 5-fold, C = 1, k = 40)

Quy tắc đặt trước: dùng `"balanced"` như Lab 03, trừ khi một mức khác có AP trung bình cao hơn `"balanced"` **nhiều hơn độ lệch chuẩn AP giữa các fold của `"balanced"`**.

| class_weight | AP | Recall | Precision | F1 | Balanced Acc. | BER |
|---|---|---|---|---|---|---|
| `None` | 0,152 ± 0,036 | 0,048 ± 0,044 | 0,132 ± 0,127 | 0,070 ± 0,065 | 0,516 ± 0,020 | 0,484 ± 0,020 |
| **`balanced`** (được chọn) | 0,163 ± 0,029 | 0,458 ± 0,061 | 0,125 ± 0,021 | 0,196 ± 0,031 | 0,614 ± 0,036 | 0,386 ± 0,036 |
| `{0:1, 1:10}` | 0,165 ± 0,032 | 0,410 ± 0,042 | 0,146 ± 0,020 | 0,214 ± 0,026 | 0,619 ± 0,023 | 0,382 ± 0,023 |

Mức AP cao nhất là `{0:1, 1:10}` (0,165), chỉ hơn `balanced` 0,002, nhỏ hơn nhiều so với std 0,029 → **giữ `balanced`**. AP gần như không đổi theo class weight; khác biệt thật nằm ở nhãn ở ngưỡng 0,5 (`None` hầu như không gắn cờ Fail).

### 3.4. Chọn mô hình cuối (LR hay RF)

Quy tắc đặt trước: **AP trung bình CV cao nhất**; bằng điểm thì chọn LR.

| Mô hình (ngưỡng 0,5, CV 5-fold trên train) | AP | Recall | Precision | F1 | Balanced Acc. | BER |
|---|---|---|---|---|---|---|
| LR (`balanced`) | 0,163 ± 0,029 | 0,458 ± 0,061 | 0,125 ± 0,021 | 0,196 ± 0,031 | 0,614 ± 0,036 | 0,386 ± 0,036 |
| RF (`balanced`) | **0,204 ± 0,070** | 0 | 0 | 0 | 0,500 | 0,500 |

→ Chọn **Random Forest** làm `model_final`. Nhưng chênh AP là 0,041, **nhỏ hơn độ lệch chuẩn giữa các fold (0,070)**, nên không khẳng định RF hơn chắc chắn (notebook 03 cũng in cảnh báo này). Ở ngưỡng 0,5 RF không gắn cờ quan sát nào (Recall = 0) trong khi LR có gắn cờ: AP đo khả năng xếp hạng, còn nhãn ở 0,5 là việc khác, nên ngưỡng riêng ở 3.5 là bước quyết định.

### 3.5. Chọn ngưỡng cho model cuối (RF) từ dự đoán out-of-fold

- **OOF:** `cross_val_predict(..., method="predict_proba")` trên **train** với cùng bộ fold; mỗi mẫu train có đúng một điểm số (1.253 điểm, 83 Fail, 64 giá trị phân biệt, lớn nhất 0,47, trung bình 0,062).
- **Quét ngưỡng** bằng vòng `for` trên 64 giá trị điểm số phân biệt; dự đoán Fail khi `score >= threshold`.
- **Quy tắc chọn:** F1 của Fail cao nhất; F1 bằng nhau thì chọn ngưỡng cao hơn (hàm `choose_threshold` trong `secom_utils.py`, khớp bảng ngưỡng). Chỉ có 1 ngưỡng đạt F1 cao nhất.
- **Hai dòng tham chiếu** (không tham gia việc chọn): "báo tất cả" và "luôn Pass".

| Chính sách (OOF của train) | Ngưỡng | Precision | Recall | F1 | Tỷ lệ cảnh báo | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| **Ngưỡng được chọn** | **0,14** | 0,187 | 0,301 | **0,230** | 10,7% (134/1.253) | 25 | 109 | 58 | 1.061 |
| Tham chiếu: báo tất cả | 0,00 | 0,066 | 1,000 | 0,124 | 100% | 83 | 1.170 | 0 | 0 |
| Tham chiếu: luôn Pass | > 0,47 | 0 | 0 | 0 | 0% | 0 | 0 | 83 | 1.170 |

**Các ngưỡng có F1 cao nhất kế tiếp:** 0,155 (F1 0,222; Precision 0,198; Recall 0,253; cảnh báo 8,5%), 0,07 (F1 0,216; Recall 0,639; cảnh báo 32,6%), 0,065 (F1 0,215; Recall 0,687; cảnh báo 35,8%). Đỉnh F1 không sắc: hạ ngưỡng xuống 0,07 đổi Recall từ 0,30 lên 0,64 nhưng phải gắn cờ gần 1/3 số quan sát.

**Chính sách đã chọn tốt hơn hai mốc không?** Có theo F1 (0,230 so với 0,124 và 0) và Balanced Accuracy (0,604 so với 0,5), nhưng ở mức khiêm tốn (Recall 0,30, Precision 0,19). Ngưỡng được chọn **trên chính OOF** của 83 Fail nên có thể lạc quan và dao động.

### 3.6. Ngưỡng cho LR từ OOF (03 mục 8.1; thêm sau lần xem test đầu tiên)

- **Lý do thêm:** trước đây chỉ RF có ngưỡng chốt, LR chỉ có 0,5, nên so sánh LR với RF chưa cùng điều kiện. Ngưỡng này được thêm **sau lần xem test đầu tiên**; nó chọn **chỉ từ OOF-train** (không dùng test) và không đổi `model_final`, quy tắc chọn mô hình hay ngưỡng RF.
- **Cách làm (đúng quy tắc của RF):** OOF của pipeline LR (`class_weight="balanced"`, k = 40, C = 1,0) bằng `cross_val_predict(..., method="predict_proba")` với cùng bộ fold, lấy cột lớp Fail theo vị trí trong `classes_`; quét ngưỡng bằng vòng `for` trên 1.253 điểm số phân biệt; chọn F1 của Fail cao nhất, bằng F1 thì chọn ngưỡng cao hơn (khớp `choose_threshold`; chỉ 1 ngưỡng đạt F1 cao nhất). Điều kiện dừng: nếu F1 không cao hơn cả hai mốc tham chiếu thì dừng, không đổi quy tắc (điều kiện này **được thoả**).

| Chính sách (OOF của train) | Ngưỡng | Precision | Recall | F1 | Tỷ lệ cảnh báo | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| **LR @ ngưỡng OOF** | **0,836** | 0,230 | 0,241 | **0,235** | 6,9% (87/1.253) | 20 | 67 | 63 | 1.103 |
| RF @ ngưỡng đã chọn | 0,14 | 0,187 | 0,301 | 0,230 | 10,7% (134/1.253) | 25 | 109 | 58 | 1.061 |
| Tham chiếu LR: báo tất cả | 0,00 | 0,066 | 1,000 | 0,124 | 100% | 83 | 1.170 | 0 | 0 |
| Tham chiếu LR: luôn Pass | > 1,0 | 0 | 0 | 0 | 0% | 0 | 0 | 83 | 1.170 |

- Chênh F1 giữa LR và RF chỉ 0,005 (vài ca trên 83 Fail), chưa đủ để nói chính sách nào hơn. Đỉnh F1 của LR khá phẳng: 10 ngưỡng đứng đầu có F1 từ 0,231 đến 0,235, nằm trong khoảng 0,69–0,84; ví dụ ngưỡng 0,69 cho Recall 0,361 nhưng gắn cờ khoảng 14,0% quan sát.
- Ngưỡng cao (0,836) là do `class_weight="balanced"` đẩy điểm số lên (điểm OOF: trung bình 0,33, trung vị 0,246, phân vị 75% = 0,493), nên điểm số này là risk score chứ không phải xác suất.
- **Hạn chế:** OOF chỉ có 83 Fail và ngưỡng được chọn trên chính OOF này nên có thể lạc quan và dao động. Lưu `artifacts/results/threshold_oof_lr.csv` và `figures/fig_03_threshold_oof_lr.png`.

### 3.7. Refit và artifact (03 mục 10)

- Refit cả hai mô hình trên toàn bộ train: LR khoảng 0,04–0,06 s, RF khoảng 0,21–0,27 s (một lần fit; số chính xác trong `decision_config.json`, dao động giữa các lần chạy).
- `decision_config.json` giữ: `model = "random_forest"`, `threshold = 0.14`, `class_weight = "balanced"`, `k = 40`, thời gian train từng mô hình, 40 cột gốc đã chọn, và 2 khoá thêm sau: `threshold_lr = 0.8361...` và `threshold_rule = "F1 Fail cao nhất trên OOF-train; bằng F1 thì chọn ngưỡng cao hơn"`. **Quy tắc chọn mô hình không được ghi trong file này**, chỉ nằm ở 03 mục 7.
- Kiểm tra: nạp lại `model_final.joblib` cho điểm số giống hệt (sai khác 0,0 trên 20 mẫu train); model nhận thẳng **590 cột thô**; model cuối chọn đúng 40 cột như mục 3.
- Score Fail có thể chưa phải xác suất đã hiệu chỉnh (do class weight); không calibrate, nên gọi là *risk score*.

---

## 4. Kết quả đánh giá trên test (notebook 04)

Test (314 quan sát, 21 Fail, 293 Pass) chỉ được dùng **một lần**: không fit, không tuning, không đổi ngưỡng; LR và RF là đối chứng xác định trước, **không chọn lại mô hình theo điểm test**. Tiêu chí chấp nhận được viết trước kết quả (markdown ở đầu notebook 04).

### 4.1. Model-level (`test_model_metrics.csv`)

| | LR @ 0,5 | RF @ 0,5 | **Model cuối (RF) @ 0,14** | LR @ ngưỡng OOF 0,836 (thêm sau) |
|---|---|---|---|---|
| TP / 21 Fail | 13 | 0 | **8** | 2 |
| FN | 8 | 21 | 13 | 19 |
| FP / 293 Pass | 76 | 0 | **26** | 14 |
| TN | 217 | 293 | 267 | 279 |
| Accuracy | 0,733 | 0,933 | 0,876 | 0,895 |
| Precision | 0,146 | 0 | 0,235 | 0,125 |
| Recall | 0,619 | 0 | 0,381 | 0,095 |
| F1 | 0,236 | 0 | 0,291 | 0,108 |
| ROC-AUC | 0,726 | 0,749 | 0,749 | 0,726 |
| Average Precision | 0,176 | 0,210 | 0,210 | 0,176 |
| Balanced Accuracy | 0,680 | 0,500 | 0,646 | 0,524 |
| BER | 0,320 | 0,500 | 0,354 | 0,476 |

- Mốc ngẫu nhiên của AP là tỷ lệ Fail: 0,0662 (train), 0,0669 (test). AP của model cuối (0,210) gấp khoảng 3,1 lần mốc; của LR (0,176) gấp khoảng 2,6 lần.
- AP và ROC-AUC không phụ thuộc ngưỡng nên RF @ 0,5 và RF @ 0,14 có cùng giá trị; LR @ 0,5 và LR @ 0,836 cũng vậy.
- **Dòng LR @ ngưỡng OOF** được thêm sau lần xem test đầu tiên (ngưỡng đọc từ `decision_config.json`, chọn từ OOF-train, không chọn bằng điểm test; `test_model_metrics.csv` giờ có 4 dòng, 3 dòng cũ giữ nguyên). Ở ngưỡng này LR gắn cờ 16/314 quan sát (5,1%) và bắt 2/21 Fail, thấp hơn Recall OOF (0,241, tương ứng khoảng 5 ca nếu giữ tỷ lệ). Với 21 Fail, chênh vài ca chưa đủ để nói ngưỡng này tệ hay tốt; một nguyên nhân có thể (chưa kiểm chứng) là điểm số của model refit trên toàn train phân bố khác điểm OOF của các model fold. Không chỉnh ngưỡng theo test. Bảng không có dòng Dummy (chưa từng có trong `test_model_metrics.csv`); "luôn đoán Pass" chỉ được nêu bằng lời (Accuracy 293/314 = 0,933).
- AP test của model cuối (0,210) gần AP trong CV (0,204 ± 0,070); F1 test (0,291) cao hơn F1 OOF (0,230) nhưng chênh cỡ này nằm trong dao động vì chỉ có 21 Fail.
- **Với 21 Fail, mỗi ca Fail đúng hay sai làm Recall đổi 4,76 điểm phần trăm.** Không có khoảng tin cậy; chênh lệch vài ca giữa các mô hình chưa đủ để kết luận mô hình nào tốt hơn.

### 4.2. System-level (`test_system_metrics.csv`, `test_robustness.csv`)

Cột chính áp dụng cho model cuối (RF @ 0,14); cột LR @ ngưỡng OOF được thêm sau. Robustness chỉ đo cho model cuối, **với LR không làm**.

| Chỉ số | RF @ 0,14 (model cuối) | LR @ ngưỡng OOF 0,836 |
|---|---|---|
| FPR | 0,089 (26/293) | 0,048 (14/293) |
| FNR | 0,619 (13/21) | 0,905 (19/21) |
| Tỷ lệ cảnh báo | 10,8% (34/314), khoảng 108 cảnh báo trên 1.000 quan sát | 5,1% (16/314), khoảng 51 trên 1.000 |
| Cảnh báo giả trên 1.000 quan sát | khoảng 83 | khoảng 45 |
| Latency median, một quan sát (toàn pipeline từ 590 cột thô, warm-up, 100 lần) | khoảng 28–30 ms với `n_jobs=-1`; khoảng 5–8 ms với `n_jobs=1` (bản sao, điểm số giống hệt) | khoảng 0,7 ms |
| Latency batch toàn test | khoảng 0,09–0,13 ms/quan sát | không đo |
| Thời gian train | khoảng 0,21–0,27 s | khoảng 0,04–0,06 s |
| Kích thước file | `model_final.joblib` = `model_rf.joblib` = 2.604,7 KB | `model_lr.joblib` = 35,1 KB |

Latency đo trên một máy phát triển (Python 3.14.3, scikit-learn 1.8.0, 16 luồng CPU), không tính thời gian nạp joblib; không suy ra latency nhà máy. Latency và thời gian train dao động giữa các lần chạy nên ghi dưới dạng khoảng; thời gian train và kích thước file của LR lấy từ artifact có sẵn.

**Robustness** (model và ngưỡng đóng băng; 3 seed; chênh so với test sạch: Recall 0,381, AP 0,210, BER 0,354, tỷ lệ cảnh báo 10,8%):

| Kịch bản | Tỷ lệ thiếu thực tế sau che | Recall (mean ± std) | Δ Recall | Δ AP | Δ BER | Δ tỷ lệ cảnh báo |
|---|---|---|---|---|---|---|
| Nhiễu Gaussian 0,05σ | 0,044 | 0,381 ± 0,000 | 0,000 | +0,007 | −0,008 | −0,015 |
| Nhiễu Gaussian 0,10σ | 0,044 | 0,318 ± 0,022 | −0,064 | −0,007 | +0,020 | −0,026 |
| Che thêm 5% ô | 0,092 | 0,365 ± 0,045 | −0,016 | −0,003 | +0,006 | −0,004 |
| Che thêm 10% ô | 0,140 | 0,333 ± 0,078 | −0,048 | +0,000 | +0,025 | −0,001 |

Nhiễu sigma = 0,05 và 0,10 × độ lệch chuẩn từng cột tính từ train thô, chỉ cộng vào ô đang quan sát; seed 42, 43, 44. Suy giảm nhẹ, cùng cỡ 1–2 ca Fail; đây là biến đổi giả lập, không đại diện đầy đủ cho sensor drift thật.

### 4.3. Industrial-level

| Chỉ số | RF @ 0,14: số đếm (quy đổi /1.000) | LR @ ngưỡng OOF: số đếm (quy đổi /1.000) |
|---|---|---|
| Fail có thể được chặn (TP), nếu mọi cảnh báo được kiểm tra và xác nhận đúng | 8/21 (25,5) | 2/21 (6,4) |
| Fail lọt (FN) | 13/21 (41,4) | 19/21 (60,5) |
| Pass cần kiểm tra thêm (FP) | 26/293 (82,8) | 14/293 (44,6) |
| Tải kiểm tra (TP + FP) | 34/314 (108,3) | 16/314 (51,0) |
| Số lần kiểm tra trung bình trên mỗi Fail chặn được = (TP + FP) / TP | 4,25 | 8,0 (chỉ dựa trên 2 ca TP, rất không ổn định) |

Quy đổi trên 1.000 quan sát là ngoại suy minh hoạ theo tỷ lệ test. Không báo số tiền tiết kiệm hay % giảm downtime vì không có dữ liệu chi phí. Khi TP = 0 thì "số lần kiểm tra trên mỗi Fail chặn được" ghi là không tính được (không xảy ra ở hai chính sách này).

### 4.4. So sánh LR @ ngưỡng OOF và RF @ 0,14 (dùng cho Chương 5; `test_lr_vs_rf.csv`)

| Chỉ số | LR @ ngưỡng OOF 0,836 | RF @ 0,14 |
|---|---|---|
| TP / FN (trên 21 Fail) | 2 / 19 | 8 / 13 |
| FP / TN (trên 293 Pass) | 14 / 279 | 26 / 267 |
| Precision / Recall / F1 | 0,125 / 0,095 / 0,108 | 0,235 / 0,381 / 0,291 |
| ROC-AUC / AP | 0,726 / 0,176 | 0,749 / 0,210 |
| Balanced Accuracy / BER | 0,524 / 0,476 | 0,646 / 0,354 |
| Tỷ lệ cảnh báo | 5,1% | 10,8% |
| Lần kiểm tra trên mỗi Fail chặn được | 8,0 | 4,25 |
| Latency một quan sát / thời gian train / kích thước | khoảng 0,7 ms / khoảng 0,04 s / 35 KB | khoảng 28 ms / khoảng 0,21 s / 2.605 KB |

- **Xếp hạng (không phụ thuộc ngưỡng):** AP test của RF cao hơn LR 0,034 và ROC-AUC cao hơn 0,022. Chênh AP **nhỏ hơn** độ lệch chuẩn AP giữa các fold trong CV của RF (0,070) và chỉ cùng cỡ của LR (0,029). Dao động do test chỉ có 21 Fail, ước lượng bằng bootstrap cặp trên dòng test (1.000 lần, seed 42): độ lệch chuẩn của chênh AP là 0,074 (khoảng 2,5%–97,5%: −0,108 đến +0,189), của chênh ROC-AUC là 0,054 (−0,090 đến +0,122); RF cao hơn trong khoảng 70% (AP) và 66% (ROC-AUC) số mẫu. Cả hai chênh lệch đều nhỏ hơn dao động này. Bootstrap chỉ đo dao động, không phải khoảng tin cậy để kết luận.
- **Điểm vận hành:** hai mô hình ở hai điểm khác nhau (LR gắn cờ 16, RF gắn cờ 34 quan sát), nên không so Recall/Precision như hai mô hình cùng một ngưỡng. RF @ 0,14 bắt nhiều Fail hơn (8 so với 2) với nhiều cảnh báo hơn; điểm của LR còn thay đổi mạnh theo ngưỡng (LR @ 0,5 gắn cờ 89 quan sát và bắt 13/21).
- **Kết luận:** vì chênh AP/ROC-AUC nằm trong dao động của 21 Fail và hai điểm vận hành khác nhau, **không kết luận mô hình nào "thắng"**. Ngưỡng LR được thêm sau lần xem test đầu tiên; model cuối vẫn là RF theo quy tắc AP CV đặt trước ở 03.

### 4.5. Tiêu chí chấp nhận và verdict

| Tiêu chí | Mục tiêu đặt trước | Kết quả đo | Kết luận |
|---|---|---|---|
| Kỹ thuật: Average Precision | > tỷ lệ Fail của train (0,0662) | 0,2102 | Đạt |
| Kỹ thuật: Balanced Accuracy | > 0,5 | 0,6461 | Đạt |
| Vận hành: latency một quan sát | chưa đặt mục tiêu số | khoảng 28–30 ms | Chưa đủ dữ liệu |
| Vận hành: tải cảnh báo | chưa đặt mục tiêu số | 10,8%, FPR 0,089 | Chưa đủ dữ liệu |
| Vận hành: robustness | chưa đặt mục tiêu số | AP đổi từ −0,007 đến +0,007 | Chưa đủ dữ liệu |
| Công nghiệp: Fail chặn được/lọt | chỉ báo cáo | TP 8/21, FN 13/21, FP 26/293 | Chưa đủ dữ liệu |

**Verdict cho prototype: Improve** (áp dụng cho model cuối RF; **không đổi** khi thêm ngưỡng LR). Tiêu chí kỹ thuật đạt; các tiêu chí vận hành và công nghiệp không có mục tiêu số nên chưa đủ dữ liệu. **Chưa đủ bằng chứng chấp nhận triển khai nhà máy.**

### 4.6. Phân tích lỗi (`test_predictions.csv`, model cuối)

- **13 Fail bị bỏ sót** có score 0,02–0,12 (đều dưới ngưỡng 0,14); **26 Pass bị cảnh báo nhầm** có score đến 0,39.
- Các ca lỗi trải từ 27/07 đến 03/10/2008. Số ô thiếu trung vị theo nhóm: FN 24, FP 28, TN 24, TP 26 (không thấy khác biệt rõ; nhóm FN 13 và TP 8 quá nhỏ để kết luận).
- Không suy ra cảm biến hay nguyên nhân vật lý từ vài quan sát này.

### 4.7. Diễn giải (cột gốc 0–589; không phải quan hệ nhân quả)

- **Top 10 RF (importance):** 103 (0,067), 59 (0,055), 455 (0,043), 21 (0,036), 65 (0,034), 64 (0,033), 213 (0,032), 247 (0,032), 130 (0,031), 129 (0,030).
- **Top 10 LR (hệ số có dấu, cột đã chuẩn hoá, dương = tăng khả năng Fail):** 436 (−2,44), 431 (+1,46), 434 (+1,03), 121 (+1,02), 64 (+0,93), 348 (+0,75), 65 (−0,74), 124 (−0,71), 160 (−0,68), 59 (+0,64).
- Trùng nhau: top 10 trùng 3 cột (59, 64, 65); top 15 trùng 6 cột. Cột có hệ số LR lớn nhất (436) không nằm trong top 15 của RF.
- Importance của RF bị chia nhỏ giữa các cột tương quan cao; hệ số LR phụ thuộc mã nhãn 1 = Fail.

---

## 5. Hạn chế cần nói thẳng

1. **Chỉ 21 Fail trong test** (83 trong train): mọi kết luận về chênh lệch giữa LR và RF đều không chắc chắn; không có khoảng tin cậy.
2. **Chênh AP giữa RF và LR trong CV (0,041) nhỏ hơn std (0,070)**; RF được chọn theo quy tắc đặt trước, không phải vì hơn chắc chắn.
3. **Ngưỡng 0,14 (RF) và 0,836 (LR) chọn trên chính OOF của 83 Fail** nên có thể lạc quan; đỉnh F1 không sắc. Ngưỡng LR được thêm sau lần xem test đầu tiên (chỉ để so sánh cùng điều kiện) và trên test chỉ bắt 2/21 Fail; hai mô hình ở hai điểm vận hành khác nhau nên Recall/Precision không so trực tiếp được.
4. **Kết quả khiêm tốn:** model cuối bắt 8/21 Fail, khoảng 1 trong 4 cảnh báo là đúng (Precision 0,235), bỏ sót 13/21 Fail.
5. **Random split** trong khi tỷ lệ Fail và missing đổi theo tuần (notebook 01): kết quả có thể lạc quan hơn khi dự đoán dữ liệu tương lai; chưa kiểm tra theo thời gian.
6. **SECOM không có** tên cảm biến, đơn vị, định nghĩa Fail, thời điểm gán nhãn, chi phí hay downtime: không gán ý nghĩa vật lý cho cột, không báo lợi ích kinh tế.
7. **Outlier được giữ nguyên** nên `StandardScaler` bị giá trị cực trị kéo (\|z\| lớn nhất 35,38 trên train).
8. Score Fail chưa hiệu chỉnh xác suất.

---

## 6. Danh sách artifact và nguồn

| File | Nội dung | Tạo ở |
|---|---|---|
| `artifacts/preprocessing_pipeline.joblib` | Pipeline tiền xử lý 4 bước, fit trên train | 02 |
| `artifacts/model_lr.joblib`, `model_rf.joblib`, `model_final.joblib` | Pipeline đầy đủ (tiền xử lý + F-test + mô hình), nhận 590 cột thô | 03 |
| `artifacts/decision_config.json` | model, ngưỡng RF, class weight, k, thời gian train, 40 cột gốc, `threshold_lr`, `threshold_rule` | 03 |
| `artifacts/results/cv_model_comparison.csv`, `cv_class_weight.csv` | CV mean/std trên train | 03 |
| `artifacts/results/oof_predictions.csv`, `threshold_oof.csv` | Điểm OOF và bảng ngưỡng của RF (kèm hai dòng tham chiếu) | 03 |
| `artifacts/results/threshold_oof_lr.csv` | Bảng ngưỡng OOF của LR (kèm hai dòng tham chiếu) | 03 |
| `artifacts/results/test_model_metrics.csv`, `test_predictions.csv`, `test_system_metrics.csv`, `test_robustness.csv` | Kết quả test (`test_model_metrics.csv` có thêm dòng LR @ ngưỡng OOF) | 04 |
| `artifacts/results/test_lr_vs_rf.csv` | Bảng so sánh gọn LR @ ngưỡng OOF và RF @ 0,14 (cho Chương 5) | 04 |
| `figures/fig_02_*.png`, `fig_03_*.png`, `fig_04_*.png` | Hình của từng notebook | 02, 03, 04 |

Cách đọc nguồn: số của mục 1 lấy từ output của `02_preprocessing.ipynb`; mục 2–3 từ `03_modeling.ipynb` và các CSV `cv_*`, `oof_predictions`, `threshold_oof`; mục 4 từ `04_evaluation.ipynb` và các CSV `test_*`. Thời gian chạy và latency dao động nhẹ giữa các lần chạy nên được ghi dưới dạng khoảng.
