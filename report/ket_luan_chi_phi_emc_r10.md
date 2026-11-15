# Hướng đi và kết quả chốt để viết báo cáo

Cập nhật ngày 15/11/2026. Nguồn: notebook [03b](../03b_cost_sensitive_modeling.ipynb), [bảng nested CV](../artifacts/cost_sensitive/nested_summary.csv) và [bảng lựa chọn cuối](../artifacts/cost_sensitive/final_policies.csv).

## 1. Bài toán và quyết định vận hành

Phân loại sản phẩm Pass/Fail từ dữ liệu SECOM; Fail = 1, Pass = 0. Timestamp không được đưa vào mô hình. Sản phẩm dự đoán Pass được cho qua; dự đoán Fail bị loại/tiêu hủy ngay, không kiểm tra lại.

Mục tiêu là giảm thiệt hại do phân loại sai, với **chi phí giả định**:

| Trường hợp | Ý nghĩa | Chi phí |
|---|---|---:|
| TN | Sản phẩm tốt được cho qua đúng | 0 |
| TP | Sản phẩm lỗi được loại đúng | 0 |
| FP | Sản phẩm tốt bị loại nhầm | 1 |
| FN | Sản phẩm lỗi bị cho qua | 10 |

\[
EMC=\frac{FP+10FN}{N},\qquad N=TP+FP+FN+TN.
\]

Một đơn vị là thiệt hại giả định của một FP. Hệ số 10 là giả định tổn thất của một FN gấp 10 lần một FP, chưa phải dữ liệu chi phí doanh nghiệp. TP/TN bằng 0 vì hàm chỉ tính thiệt hại do quyết định sai, không cộng chi phí sản xuất vốn có. Trong notebook, kịch bản tương ứng là **`emc_r10`**.

## 2. Phương pháp thực hiện trong 03b

- Dữ liệu: 1.567 sản phẩm, 590 đặc trưng. Chia stratified 80/20, seed 42.
- Train: **1.253 mẫu, 1.170 Pass và 83 Fail**. Holdout: **314 mẫu, 293 Pass và 21 Fail**; 03b không dùng holdout để lựa chọn hoặc đánh giá mới.
- Pipeline: MissingRateSelector(0.70) → SimpleImputer(median) → VarianceThreshold(0) → StandardScaler → SelectKBest(f_classif) → classifier. Mọi bước được fit trong phần train của mỗi fold.
- Thử **9 cấu hình LR và 12 cấu hình RF**, đều dùng `class_weight="balanced"`.
- Trong vòng lựa chọn, lấy score OOF cho từng cấu hình. Với mỗi ngưỡng, tính FP, FN và EMC. Chọn ngưỡng rẻ nhất của từng cấu hình, sau đó chọn cấu hình rẻ nhất. Nếu hòa chi phí, ưu tiên ít gán Fail hơn, sau đó ngưỡng cao hơn, rồi candidate ID khi vẫn hòa.
- Dùng **nested CV: 5 fold ngoài, 3 fold trong** để đánh giá quy trình lựa chọn. Vòng trong chọn cấu hình/ngưỡng; vòng ngoài giữ nguyên lựa chọn để đánh giá trên phần chưa tham gia lựa chọn.
- Cuối cùng, dùng OOF 5-fold trên toàn bộ train để chọn policy cuối và fit lại pipeline trên train. Điểm ở bước này dùng để lựa chọn, không thay cho kết quả nested CV.

Phần tìm ngưỡng được viết basic bằng `for` tại mục **10.1 của 03b**; ô này tạo lại OOF thật cho một cấu hình và hiển thị 10 dòng có chi phí thấp nhất.

## 3. Kết quả đánh giá để đưa vào báo cáo

Các dòng sau lấy từ `nested_summary.csv`, lọc **`scenario = emc_r10`**. Đây là số đếm gộp dự đoán vòng ngoài trên 1.253 mẫu train, không phải kết quả test trên 314 mẫu.

| Quy trình | TP | FP | FN | TN | EMC ↓ | Precision | Recall Fail | F1 | BER ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR tinh chỉnh riêng | 40 | 277 | 43 | 893 | 0,564246 | 0,126183 | 0,481928 | 0,200000 | 0,377412 |
| RF tinh chỉnh riêng | 38 | 246 | 45 | 924 | 0,555467 | 0,133803 | 0,457831 | 0,207084 | 0,376213 |
| Chọn LR/RF theo chi phí (`cost_selected`) | 38 | 246 | 45 | 924 | **0,555467** | 0,133803 | 0,457831 | 0,207084 | 0,376213 |
| Luôn Pass | 0 | 0 | 83 | 1.170 | 0,662410 | 0 | 0 | 0 | 0,5 |
| Luôn Fail | 83 | 1.170 | 0 | 0 | 0,933759 | 0,066241 | 1 | 0,124251 | 0,5 |

Với `cost_selected`:

\[
EMC=\frac{246+10\times45}{1253}=\frac{696}{1253}\approx0,555467.
\]

Tương đương **555,466879 đơn vị chi phí trên 1.000 sản phẩm** theo tỷ lệ lớp đang đánh giá. So với luôn Pass, tổng chi phí giả định giảm từ 830 xuống 696, tức khoảng **16,14%**. Đây là mức giảm quan sát trong CV theo giả định, chưa phải mức tiết kiệm tiền thực tế.

RF có chi phí quan sát thấp hơn LR, nhưng Recall thấp hơn: bỏ sót 45/83 lỗi so với 43/83 của LR, đổi lại ít loại nhầm hơn: 246 so với 277. Mức bỏ sót này cần được đối chiếu yêu cầu doanh nghiệp trước khi kết luận triển khai. Hiện chưa có Recall tối thiểu hoặc giới hạn loại nhầm được cung cấp.

## 4. Ngưỡng cuối được chọn

Lấy từ `final_policies.csv`, lọc `scenario = emc_r10` và `selection_scope = cost_selected`:

| Thuộc tính | Giá trị |
|---|---|
| Candidate ID | `rf_k80_dNone_leaf5` |
| Mô hình | RandomForestClassifier |
| Số cây | 200 |
| Độ sâu | `None` — không giới hạn |
| Số mẫu tối thiểu mỗi lá | 5 |
| Trọng số lớp | `balanced` |
| SelectKBest | 80 đặc trưng |
| Ngưỡng | **0,344288** |

Dùng đúng pipeline đã fit: **score Fail ≥ 0,344288 → Fail/loại; score thấp hơn → Pass/cho qua**. Ngưỡng này gắn với pipeline 03b đã chọn, không áp chung cho LR hoặc RF trong 03c. Score chưa được xác nhận là xác suất đã hiệu chỉnh.

Trên OOF dùng để chọn cuối: TP = 36, FP = 152, FN = 47, TN = 1.018. EMC = `(152 + 10 × 47) / 1253` = **0,496409**. Đây là điểm lựa chọn, có thể lạc quan. **Dùng 0,555467 từ nested CV khi báo cáo hiệu quả quy trình.** Không gắn ngưỡng cuối 0,344288 cho toàn bộ dự đoán vòng ngoài, vì mỗi fold ngoài được đánh giá với policy chọn từ vòng trong của riêng fold đó.

## 5. Vai trò của 03c và hướng mở rộng

03c đã chạy ba mô hình XGBoost, LR L1 và RF balanced qua năm seed. Cấu hình trong từng họ được chọn theo OOF AP; ngưỡng được chọn theo công thức cũ `(TP + FP + 10FN) / N`. Kết quả đó vẫn là số đo thực nghiệm của thử nghiệm cũ.

Hướng mở rộng là giữ preprocessing bốn bước và ba mô hình của 03c, chọn cấu hình/ngưỡng theo EMC mới, rồi đánh giá quy trình bằng nested CV. **Thử nghiệm ba mô hình theo hướng mới chưa chạy**, nên chưa có cơ sở đưa ra bảng kết quả mới hoặc kết luận mô hình nào tốt nhất dưới quy trình đó.

## 6. Đoạn kết luận có thể dùng trong báo cáo

> Với giả định thiệt hại do loại nhầm một sản phẩm tốt là 1 đơn vị và bỏ sót một sản phẩm lỗi là 10 đơn vị, nhóm tối thiểu hóa EMC = (FP + 10FN)/N. Cấu hình và ngưỡng được lựa chọn bằng dự đoán OOF, còn hiệu quả của toàn bộ quy trình được đánh giá bằng nested CV 5 fold ngoài và 3 fold trong trên 1.253 mẫu thuộc phần train ban đầu. Quy trình chọn LR/RF theo chi phí đạt EMC 0,555467 đơn vị/sản phẩm, giảm khoảng 16,14% so với phương án luôn Pass theo cùng giả định. Tuy nhiên, quy trình vẫn bỏ sót 45/83 sản phẩm lỗi. Policy cuối được chọn trên OOF toàn train là Random Forest 200 cây, chọn 80 đặc trưng, ngưỡng 0,344288. Các kết quả phản ánh hiệu quả thực nghiệm dưới ma trận chi phí giả định, chưa chứng minh mức tiết kiệm thực tế hoặc đáp ứng yêu cầu triển khai doanh nghiệp.

## 7. Tái lập và nguồn số liệu

- [Hướng dẫn chạy 03b](../HUONG_DAN_03B_COST_SENSITIVE.md), [quy ước chi phí](../QUY_UOC_CHI_PHI_DA_CHOT.md).
- [nested_summary.csv](../artifacts/cost_sensitive/nested_summary.csv): dùng để báo cáo chất lượng.
- [final_policies.csv](../artifacts/cost_sensitive/final_policies.csv): cấu hình/ngưỡng cuối và điểm lựa chọn.
- [run_metadata.json](../artifacts/cost_sensitive/run_metadata.json): seed, phiên bản, giả định và cấu hình đã chạy.
- Dùng đúng kernel Python 3.11 đã cài các phiên bản trong metadata. Khác phiên bản thư viện có thể làm RF khác kết quả.
- Raw SECOM và model nhị phân không nằm trong bản cập nhật này. Notebook có output nhúng; chạy lại để tạo model và các bảng chi tiết còn lại.
