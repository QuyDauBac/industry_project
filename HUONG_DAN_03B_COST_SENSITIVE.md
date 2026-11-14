# Hướng dẫn notebook 03b: phân loại Pass/Fail theo chi phí

Tài liệu cho thành viên nhóm đọc và chạy [03b_cost_sensitive_modeling.ipynb](./03b_cost_sensitive_modeling.ipynb). Phần xử lý và huấn luyện nằm trong [secom_cost_sensitive.py](./secom_cost_sensitive.py); cách đọc dữ liệu và tính các chỉ số dùng lại từ [secom_utils.py](./secom_utils.py).

## Đọc trong 2 phút

- Bài toán là **phân loại sản phẩm Pass/Fail** từ 590 phép đo. Timestamp không phải đầu vào mô hình.
- **Kết quả modeling cũ là số đo thực nghiệm**, giữ làm baseline. Chỉ **chi phí kinh doanh** trong thử nghiệm mới là giả định.
- Notebook thử 21 cấu hình LR/RF và 6 kịch bản chi phí. Với mỗi kịch bản, nó chọn cấu hình và ngưỡng sao cho chi phí thấp nhất trên dữ liệu dùng để chọn.
- **CV vòng trong dùng để chọn; CV vòng ngoài dùng để đánh giá.** Tập test 314 mẫu không được dùng để chọn hoặc chấm điểm mới.
- `r = 20` nghĩa là **bỏ sót một sản phẩm lỗi được giả định tốn gấp 20 lần kiểm tra một sản phẩm**. Nó không có nghĩa là bỏ sót 20 sản phẩm hay bỏ sót 20%.
- Để đánh giá chất lượng, đọc `nested_summary.csv` sau khi chạy. `final_policies.csv` là kết quả chọn cấu hình cuối, có thể lạc quan vì đã được tối ưu trên chính OOF đó.

Nếu chỉ cần hiểu ý nghĩa kết quả, đọc mục **5, 8 và 10**. Nếu cần chạy lại, bắt đầu từ mục **1–2**.

## 1. Lấy đúng phiên bản từ GitHub

Các file mới đang nằm trên fork `Ductri2006/industry_project`, nhánh `cost-sensitive-modeling`, trong [PR nháp #1](https://github.com/QuyDauBac/industry_project/pull/1) gửi vào repo gốc.

**Pull nhánh `main` của repo gốc chỉ có các file này sau khi PR được merge.** Nếu muốn xem trước bằng một bản sao mới:

```powershell
git clone --branch cost-sensitive-modeling https://github.com/Ductri2006/industry_project.git industry_project_cost_sensitive
cd industry_project_cost_sensitive
```

Không cần ghi đè hoặc kéo thay đổi vào một thư mục đang có công việc chưa lưu. Có thể mở notebook trên GitHub để xem các output đã được lưu mà chưa chạy lại.

## 2. Chuẩn bị và chạy notebook

Các file cần có ở thư mục gốc:

- `03b_cost_sensitive_modeling.ipynb` và `secom_cost_sensitive.py`.
- `secom_utils.py` và `secom_preprocessing.py` đã có trong repo.
- `requirements.txt` để cài thư viện.

Đặt dữ liệu SECOM vào đúng vị trí:

```text
data/secom/secom.data
data/secom/secom_labels.data
```

Hai file `.data` được Git bỏ qua; **pull/clone repo không tự mang theo dữ liệu này**. Dùng bộ dữ liệu SECOM mà nhóm đang sử dụng. Xem thêm [README của dự án](./README.md).

Ví dụ tạo môi trường trên Windows, chạy từ thư mục gốc:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter notebook
```

Mở notebook, chọn kernel Python của môi trường vừa cài và chạy **Restart Kernel and Run All**. Notebook gốc ghi tên kernel `secom-python311`; nếu máy chưa có tên này, có thể chọn kernel `.venv` trong giao diện, hoặc đăng ký khi cần:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name secom-python311 --display-name "Python (SECOM .venv)"
```

Không cần chạy notebook 02/03 trước để tạo pipeline cho 03b: 03b tự xây pipeline từ dữ liệu thô. Các bảng modeling cũ chỉ được đọc để tham khảo nếu có. Thời gian chạy phụ thuộc máy vì notebook phải fit nhiều pipeline trong CV.

`requirements.txt` hiện chưa ghim phiên bản. Sau mỗi lần chạy, `run_metadata.json` ghi lại phiên bản thư viện, seed và cấu hình. Khác phiên bản có thể làm kết quả thay đổi; không mặc định mọi máy sẽ cho số giống tuyệt đối.

## 3. Dữ liệu và pipeline

| Tập dữ liệu | Số mẫu | Pass | Fail | Vai trò trong notebook mới |
|---|---:|---:|---:|---|
| Toàn bộ SECOM | 1.567 | 1.463 | 104 | Nạp dữ liệu và chia tập |
| Train | 1.253 | 1.170 | 83 | Phát triển, nested CV và fit cấu hình cuối |
| Test/holdout | 314 | 293 | 21 | Giữ ngoài mọi phép chọn và đánh giá mới |

Chia stratified 80/20 theo `get_split`, seed 42. Stratified giữ tỷ lệ lớp gần giống giữa các phần; không làm hai lớp có số mẫu bằng nhau. Nhãn gốc `-1/1` được đổi thành **Pass = 0, Fail = 1**.

Pipeline chạy theo thứ tự:

```text
590 phép đo thô
  → lọc cột thiếu quá nhiều
  → điền thiếu bằng trung vị
  → loại cột hằng
  → chuẩn hóa
  → chọn k đặc trưng
  → LR hoặc RF
  → score của lớp Fail
  → so với ngưỡng để ra nhãn Pass/Fail
```

| Bước | Thành phần | Cách làm và mục đích |
|---|---|---|
| 1 | `MissingRateSelector(0.70)` | Tính tỷ lệ NaN từng cột; loại cột thiếu **hơn 70%**. |
| 2 | `SimpleImputer(strategy="median")` | Điền NaN bằng trung vị của từng cột đã học từ phần train tương ứng. |
| 3 | `VarianceThreshold(0.0)` | Loại cột không biến thiên sau khi điền thiếu. Không loại riêng cột gần hằng. |
| 4 | `StandardScaler()` | Chuẩn hóa từng cột: $z=(x-\mu)/\sigma$, với thống kê học từ phần train. |
| 5 | `SelectKBest(f_classif, k)` | Chọn k cột có ANOVA F-score cao nhất liên quan đến nhãn. Thử k = 20, 40, 80. |
| 6 | Classifier | Logistic Regression hoặc Random Forest tạo score cho lớp Fail. |

**Toàn bộ bước học từ dữ liệu đều được fit lại trong từng fold train**, kể cả điền thiếu và chọn đặc trưng. Fold dùng để đánh giá chỉ được transform và dự đoán bằng pipeline đã fit.

Số cột `590 → 582 → 466` là kết quả tiền xử lý trên toàn bộ train ở baseline cũ. Trong từng fold, số cột giữ lại có thể khác vì thống kê được học lại. Chuẩn hóa không tự xóa outlier; notebook không thêm clipping hoặc SMOTE.

## 4. Xử lý imbalance và các cấu hình được thử

Cả LR và RF dùng `class_weight="balanced"`. Trọng số của lớp c được tính trên dữ liệu đang fit:

$$
w_c=\frac{N_{fit}}{K\,N_{c,fit}}.
$$

Với K = 2, lớp Fail ít mẫu nên mỗi mẫu Fail có trọng số lớn hơn. Trên toàn bộ train, trọng số Pass khoảng 0,535 và Fail khoảng 7,548: một Fail có trọng số khoảng **14,1 lần** một Pass. Các fold có trọng số riêng.

**14,1 là kết quả từ tỷ lệ lớp, không phải tỷ lệ chi phí doanh nghiệp.** `class_weight` tác động việc học; `r` tác động cách chọn cấu hình và ngưỡng theo chi phí. Không tự coi hai đại lượng này tương đương.

Trong notebook này, đặt r = 20 **không đổi** classifier thành `class_weight={0: 1, 1: 20}`. Trọng số vẫn là `"balanced"`; chi phí được dùng để chọn tham số, họ mô hình và ngưỡng. Cùng một cấu hình có thể dùng lại điểm OOF cho nhiều kịch bản chi phí.

| Mô hình | Các tham số thay đổi | Các thiết lập giữ cố định | Số cấu hình |
|---|---|---|---:|
| LR | k = 20/40/80; C = 0,1/1/10 | balanced, `max_iter=2000` | 3 × 3 = **9** |
| RF | k = 20/40/80; depth = 5/None; leaf = 1/5 | balanced, 200 cây, `n_jobs=1` | 3 × 2 × 2 = **12** |

Tổng cộng **21 cấu hình**. C nhỏ hơn tăng mức regularization của LR; depth/leaf kiểm soát độ phức tạp của RF. Đây là phạm vi thử đã định trước, không phải khẳng định các giá trị ấy tối ưu cho mọi dữ liệu.

## 5. Ma trận nhầm lẫn và giả định chi phí

| Thực tế | Dự đoán Pass | Dự đoán Fail |
|---|---|---|
| Pass | **TN:** sản phẩm tốt được nhận đúng | **FP:** sản phẩm tốt bị gán lỗi, phải kiểm tra nhầm |
| Fail | **FN:** sản phẩm lỗi bị bỏ sót | **TP:** sản phẩm lỗi được phát hiện đúng |

N = TP + FP + FN + TN. Nếu mọi nhãn Fail đều dẫn đến kiểm tra lại, số sản phẩm cần kiểm tra là **TP + FP**, không chỉ FP.

Công thức chi phí trung bình tổng quát:

$$
EC=\frac{C_{TP}TP+C_{FP}FP+C_{FN}FN+C_{TN}TN}{N}.
$$

Notebook phân tích hai họ chi phí riêng:

| Kết quả | `emc_r5/r10/r20`: EMC chỉ tính lỗi | `review_r5/r10/r20`: kiểm tra mọi nhãn Fail |
|---|---:|---:|
| TN | 0 | 0 |
| FP | 1 | 1 |
| FN | r | r |
| TP | 0 | 1 |

EMC thông thường coi dự đoán đúng không tốn chi phí:

$$
EMC=\frac{FP+rFN}{N}.
$$

Họ `review_*` là **kịch bản chính** của bài toán nhóm đang trao đổi: cả cảnh báo đúng và nhầm đều phải kiểm tra lại:

$$
EC_{review}=\frac{TP+FP+rFN}{N}.
$$

- `total_cost`: tử số của công thức tương ứng.
- `cost_per_product`: chi phí trung bình = `total_cost / N`.
- `cost_per_1000`: `1000 × cost_per_product`, tính theo tỷ lệ lớp của dữ liệu đánh giá.

**1 ở đây là một đơn vị chi phí tương đối**, không phải 1 đồng. `r = 5, 10, 20` là các giả định về chi phí **một lần bỏ sót** so với 1 đơn vị tham chiếu. Trong họ review, đơn vị tham chiếu là một lần kiểm tra; trong họ EMC, đó là một lỗi FP.

Ví dụ riêng họ review: nếu một lần kiểm tra là 10.000 đồng, r = 20 tương ứng giả định mỗi lần bỏ sót gây tổn thất kỳ vọng 200.000 đồng. Ví dụ này không phải số liệu công ty. **r = 20 không có nghĩa bỏ sót 20 sản phẩm, hay càng đắt thì bỏ sót càng nhiều.**

Giả định hiện tại: kiểm tra lại phát hiện được lỗi; chi phí FN là tổn thất kỳ vọng còn lại khi bỏ sót tại bước này, có xét các bước QC tiếp theo. FN không tự động có nghĩa sản phẩm lỗi đã được giao cho khách hàng. Chưa đặt ràng buộc Recall tối thiểu hoặc năng lực kiểm tra tối đa vì chưa có yêu cầu định lượng từ doanh nghiệp.

## 6. Chọn ngưỡng như thế nào?

Với score s và ngưỡng t:

$$
\hat y=\begin{cases}1\;(Fail),&s\ge t\\0\;(Pass),&s<t.\end{cases}
$$

Điểm OOF của phần train dùng để chọn được quét qua các giá trị score phân biệt, thêm ngưỡng 0 và một ngưỡng cố định lớn hơn 1 cho phương án luôn Pass. Ngưỡng lớn hơn 1 là **quy ước cho hành động không gắn Fail**, không phải xác suất lớn hơn 100%.

Trong mỗi kịch bản, ưu tiên theo thứ tự: **chi phí thấp nhất → ít sản phẩm kiểm tra nhất → ngưỡng cao hơn → candidate ID** nếu vẫn hòa. Quy tắc cuối chỉ giúp kết quả xác định khi các lựa chọn tương đương.

Đổi ngưỡng thay đổi TP/FP/FN/TN, không tự làm khả năng xếp hạng tốt hơn. Score từ mô hình dùng class weight chưa được xác nhận là xác suất đã hiệu chỉnh; không hiểu score 0,2 chắc chắn là “20% khả năng lỗi”, cũng không áp thẳng công thức ngưỡng lý thuyết theo tỷ lệ chi phí. Notebook chọn ngưỡng thực nghiệm bằng CV.

## 7. CV hai vòng: chọn ở đâu, đánh giá ở đâu?

**CV (Cross-Validation)** chia dữ liệu thành nhiều phần, lần lượt dùng một phần để đánh giá mô hình học từ các phần còn lại. **OOF (out-of-fold)** là dự đoán của mẫu do mô hình không được huấn luyện trên chính mẫu đó tạo ra.

Notebook dùng **nested CV = 5 fold ngoài × 3 fold trong**, chỉ trên 1.253 mẫu train:

1. Giữ một fold ngoài làm tập đánh giá; khoảng 1.002–1.003 mẫu còn lại là phần để lựa chọn.
2. Với từng cấu hình, chạy CV trong 3 fold trên phần để lựa chọn, tạo score OOF. Mỗi lần fit đều học lại toàn bộ pipeline từ dữ liệu thô của fold train.
3. Dùng score OOF trong để chọn ngưỡng, tham số và, với `cost_selected`, cả họ LR/RF theo chi phí kịch bản.
4. Fit cấu hình đã chọn trên toàn bộ phần train của vòng ngoài. Giữ nguyên ngưỡng đã chọn và dự đoán fold ngoài một lần.
5. Lặp cho đủ 5 fold ngoài. Mỗi mẫu train có đúng một dự đoán ngoài cho mỗi kịch bản/quy trình.

Trong một fold, OOF của cấu hình được dùng lại cho nhiều kịch bản; không cần fit lại chỉ vì đổi r. **Nhãn của fold ngoài không tham gia chọn cấu hình/ngưỡng của fold đó.**

Các số TP/FP/FN/TN được cộng từ 5 fold ngoài, rồi tính lại chỉ số trên các tổng đó; không lấy trung bình các tỷ lệ đã làm tròn. `cost_per_product_fold_std` là độ lệch chuẩn chi phí/sản phẩm giữa 5 fold, không phải khoảng tin cậy và không cùng thang với chi phí/1.000 sản phẩm.

Code dùng độ lệch chuẩn mẫu (`ddof=1`): $s=\sqrt{\sum_{f=1}^{5}(EC_f-\overline{EC})^2/(5-1)}$, với $EC_f$ là chi phí trên **một** sản phẩm ở fold f. Không so trực tiếp s với `cost_per_1000` mà quên hệ số 1.000.

### Hiểu cột `selection_scope`

| Giá trị | Cách tạo kết quả |
|---|---|
| `LR` | Chỉ chọn trong 9 cấu hình LR, đồng thời chọn ngưỡng trong vòng trong. |
| `RF` | Chỉ chọn trong 12 cấu hình RF, đồng thời chọn ngưỡng trong vòng trong. |
| `cost_selected` | Vòng trong chọn trong cả 21 cấu hình LR/RF và các ngưỡng. Đây là quy trình chọn mô hình hoàn chỉnh. |
| `always_pass` | Gán tất cả là Pass: không kiểm tra, bỏ sót mọi Fail. |
| `review_all` | Gán tất cả là Fail; trong họ review tương ứng kiểm tra tất cả. |

`cost_selected` **không phải** lấy dòng LR/RF có chi phí thấp nhất sau khi đã xem fold ngoài. Vì vậy nó có thể không thấp nhất trong bảng đánh giá. Mỗi fold có thể chọn tham số, ngưỡng và họ mô hình khác nhau: kết quả gộp CV không có một ngưỡng duy nhất để mang đi dùng.

Sau phần đánh giá, notebook chạy **OOF 5-fold trên toàn bộ train** để chọn cấu hình/ngưỡng cuối cho từng kịch bản và scope LR/RF/cost_selected, rồi refit trên toàn bộ train. Có 6 × 3 = 18 policy cấu hình; nhiều policy có thể dùng chung một pipeline đã fit. Các điểm OOF cuối này mang nhãn `selection_only_not_unbiased`, dùng để lựa chọn, không thay thế đánh giá ngoài `nested_outer_evaluation`.

## 8. Công thức các chỉ số hỗ trợ

Trong bài này, lớp dương là **Fail = 1**. Dấu ↑ nghĩa là cao hơn tốt hơn; ↓ nghĩa là thấp hơn tốt hơn, khi giữ cùng mục tiêu và dữ liệu đánh giá.

| Chỉ số | Công thức | Ý nghĩa |
|---|---|---|
| Precision ↑ | $TP/(TP+FP)$ | Trong các sản phẩm được gắn Fail, bao nhiêu thực sự lỗi? |
| Recall Fail ↑ | $TP/(TP+FN)$ | Trong các sản phẩm thực sự lỗi, tìm được bao nhiêu? |
| F1 ↑ | $2TP/(2TP+FP+FN)$ | Tổng hợp Precision và Recall của Fail. |
| FPR ↓ | $FP/(FP+TN)$ | Tỷ lệ sản phẩm tốt bị gán Fail. |
| FNR ↓ | $FN/(FN+TP)$ | Tỷ lệ sản phẩm lỗi bị bỏ sót; bằng 1 − Recall. |
| Balanced Accuracy ↑ | $(Recall+TN/(TN+FP))/2$ | Trung bình tỷ lệ nhận đúng của hai lớp. |
| BER ↓ | $(FNR+FPR)/2$ | Bằng 1 − Balanced Accuracy. |
| Accuracy ↑ | $(TP+TN)/N$ | Tỷ lệ đúng tổng thể; dễ cao do Pass chiếm đa số. |
| Inspection/alarm rate | $(TP+FP)/N$ | Tỷ lệ sản phẩm được gắn Fail, phải kiểm tra trong họ review. |

Trong code, Precision/F1 được ghi là 0 khi mẫu số bằng 0. AP và ROC-AUC mô tả khả năng xếp hạng qua nhiều ngưỡng; chúng là thông tin bổ sung, không phải mục tiêu lựa chọn chính của notebook chi phí.

**Không tối ưu riêng Recall hoặc Precision:** luôn gán Fail cho Recall = 100% nhưng kiểm tra mọi sản phẩm; luôn gán Pass tránh mọi FP nhưng bỏ sót mọi Fail. Cần nhìn cả chi phí và số đếm.

## 9. Đọc đúng bảng kết quả và tệp được lưu

Sau khi chạy, các tệp sinh lại nằm trong `artifacts/cost_sensitive/`:

| Tệp | Dùng để làm gì? |
|---|---|
| `cost_scenarios.csv`, `candidate_grid.csv` | Xem chính xác chi phí và lưới cấu hình đã thử. |
| **`nested_summary.csv`** | **Bảng chính để đánh giá:** số đếm gộp và chi phí từ các fold ngoài. |
| `nested_fold_metrics.csv` | Xem kết quả và cấu hình được chọn trong từng fold ngoài. |
| `nested_outer_predictions.csv` | Kiểm tra nhãn/score/dự đoán ngoài của từng mẫu. |
| `nested_inner_search.csv` | Xem quá trình lựa chọn trong từng vòng trong; không dùng làm điểm đánh giá cuối. |
| `final_policies.csv`, `final_selection_search.csv` | Cấu hình chọn cuối và điểm OOF dùng để chọn; có thể lạc quan. |
| `policy_configs.json`, `models/` | Pipeline đã refit và ngưỡng tương ứng để dùng thử từng policy. |
| `run_metadata.json` | Seed, phiên bản thư viện, thời gian chạy, giả định và cấu hình. |

Hai đồ thị sinh vào `figures/cost_sensitive/`: chi phí theo tỷ lệ r và Recall theo tỷ lệ kiểm tra. Chỉ so chi phí LR/RF/baseline **trong cùng kịch bản**. Chi phí thô khác r hoặc khác ma trận không cùng bài toán tối ưu, nên không dùng để kết luận r nào “tốt nhất”.

Các CSV/model/PNG mới này được sinh khi chạy, không được đẩy kèm commit notebook ban đầu. Pull về vẫn xem được output nhúng trong notebook; cần chạy lại để có các tệp riêng trên máy. Không dùng thêm preprocessing bên ngoài pipeline khi dự đoán; phải giữ đúng thứ tự 590 cột và ngưỡng của policy.

## 10. Kết quả lần chạy hiện tại và cách diễn giải

Bảng sau lấy từ `nested_summary.csv`, scope **`cost_selected`**, họ chi phí **review**. N = 1.253, gồm 83 Fail và 1.170 Pass. Đây là số đo CV, không phải kết quả trên tập test 314 mẫu.

| Kịch bản | TP | FP | FN | TN | Kiểm tra TP+FP | Tổng chi phí | Chi phí/1.000 sản phẩm |
|---|---:|---:|---:|---:|---:|---:|---:|
| `review_r5` | 12 | 46 | 71 | 1.124 | 58 | 413 | 329,6089 |
| `review_r10` | 38 | 246 | 45 | 924 | 284 | 734 | 585,7941 |
| `review_r20` | 60 | 573 | 23 | 597 | 633 | 1.093 | 872,3065 |

Ví dụ **r = 20**:

$$
total\_cost=(60+573)+20\times23=633+460=1093.
$$

$$
cost\_per\_product=1093/1253\approx0.8723065;\qquad cost\_per\_1000\approx872.3065.
$$

Đọc bằng lời: phải kiểm tra 633 sản phẩm, trong đó 60 thực sự lỗi và 573 là sản phẩm tốt bị gắn Fail; còn 23/83 lỗi bị bỏ sót. Recall khoảng 72,29%, nhưng Precision chỉ khoảng 9,48%. **Giảm bỏ sót đi kèm nhiều lượt kiểm tra nhầm** trong kết quả này.

So sánh hai họ mô hình, theo chi phí/1.000 sản phẩm của cùng kịch bản review:

| Kịch bản | LR tinh chỉnh | RF tinh chỉnh |
|---|---:|---:|
| r = 5 | 335,1955 | 329,6089 |
| r = 10 | 578,6113 | 585,7941 |
| r = 20 | 907,4222 | 872,3065 |

RF có chi phí quan sát thấp hơn ở r = 5 và 20; LR thấp hơn ở r = 10. Ở r = 10, `cost_selected` vẫn bằng kết quả RF vì nó đã chọn bằng vòng trong, không chọn lại theo bảng ngoài.

Hai mốc tham chiếu trong họ review: **luôn Pass có tổng chi phí = 83r**; **kiểm tra tất cả có tổng chi phí = 1.253**. Với r = 5, quy trình chọn có chi phí 413 so với 415 của luôn Pass: lợi ích quan sát rất nhỏ. Chưa đủ để khẳng định mức tiết kiệm ổn định. Không quy đổi các chi phí giả định thành lợi nhuận thực tế.

## 11. Những điểm nhóm cần thống nhất khi báo cáo

1. **Chỉ chi phí là giả định.** Nhãn, dữ liệu và các số đo chạy mô hình không được gọi là số giả lập.
2. Chỉ số chính của thử nghiệm mới là chi phí theo từng ma trận; vẫn báo Precision, Recall, BER và TP/FP/FN/TN để người đọc thấy đánh đổi.
3. Không lấy kết quả test cũ so với nested CV mới rồi gọi phần chênh là mức cải thiện chắc chắn: chúng có cách chọn và tập đánh giá khác nhau.
4. Chưa có chi phí thật, giới hạn kiểm tra hoặc Recall tối thiểu, nên chưa có một policy doanh nghiệp duy nhất. Chi phí thấp không tự chứng minh mức bỏ sót là chấp nhận được.
5. Dữ liệu có ít Fail; dao động giữa fold và lựa chọn ngưỡng còn đáng kể. Những kết quả này hỗ trợ đánh giá prototype, chưa đủ xác nhận triển khai nhà máy.
6. Thay đổi r, lưới tham số hoặc ràng buộc là một thí nghiệm mới. Giữ lại metadata; xác nhận bằng dữ liệu mới trước khi kết luận vận hành.

### Lỗi chạy thường gặp

- **Không tìm thấy dữ liệu:** kiểm tra đúng hai tên `.data` trong `data/secom/`; các file này không có sẵn chỉ nhờ pull Git.
- **Không import được module:** mở/chạy notebook từ thư mục gốc hoặc thư mục con của dự án, giữ đủ ba file Python hỗ trợ đã nêu ở mục 2.
- **Không tìm thấy kernel:** chọn môi trường đã cài thư viện, hoặc đăng ký `secom-python311` theo mục 2; không nhầm kernel Python của môi trường khác.
- **Không thấy CSV/model/PNG mới:** chạy toàn bộ notebook để sinh lại; bản notebook trên GitHub đã có output nhúng để đọc trước.
- **Kết quả khác bảng tài liệu:** kiểm tra dữ liệu, seed, cấu hình, phiên bản trong `run_metadata.json`; bảng mục 10 là ảnh chụp kết quả của lần chạy hiện tại.

## Tài liệu tham khảo

- [scikit-learn: điều chỉnh ngưỡng theo chi phí kinh doanh](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html).
- [scikit-learn: nested và non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
- Mã thực thi của dự án: [secom_cost_sensitive.py](./secom_cost_sensitive.py), [secom_utils.py](./secom_utils.py), [secom_preprocessing.py](./secom_preprocessing.py).
