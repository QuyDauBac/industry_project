# Quy ước chi phí đã chốt

Ngày chốt: 15/11/2026. Quy ước này thay thế giả định kiểm tra lại mọi sản phẩm dự đoán Fail trong báo cáo đối chiếu trước đó.

## Quy trình doanh nghiệp dùng cho thử nghiệm

- Dự đoán Pass: cho qua.
- Dự đoán Fail: loại hoặc tiêu hủy ngay, không kiểm tra lại.
- Nhãn thực tế Pass/Fail dùng để đánh giá quyết định của mô hình.

## Bảng chi phí giả định

Quy ước Fail = 1, Pass = 0.

| Trường hợp | Ý nghĩa | Chi phí |
|---|---|---:|
| TN | Sản phẩm tốt được cho qua đúng | 0 |
| TP | Sản phẩm lỗi được loại đúng | 0 |
| FP | Sản phẩm tốt bị tiêu hủy nhầm | 1 |
| FN | Sản phẩm lỗi bị cho qua | 10 |

Chi phí tổng: `FP + 10 × FN`.

Chi phí trung bình: **`EMC = (FP + 10 × FN) / N`**, với `N = TP + TN + FP + FN`.

Một đơn vị là thiệt hại giả định khi tiêu hủy nhầm một sản phẩm tốt. FN = 10 nghĩa là thiệt hại bỏ sót một sản phẩm lỗi được giả định cao gấp 10 lần thiệt hại của một FP. Đây chưa phải số liệu chi phí thực tế của doanh nghiệp.

TP/TN bằng 0 biểu thị không cộng thêm thiệt hại do quyết định phân loại sai. Các chi phí sản xuất và loại bỏ sản phẩm vốn đã lỗi không được tính trong hàm này.

## Hướng lựa chọn mô hình và ngưỡng

**Kết quả đang chốt để báo cáo là notebook 03b, kịch bản `emc_r10`.** Chọn cấu hình và ngưỡng theo EMC thấp nhất trên OOF; đánh giá toàn bộ quy trình bằng nested CV. AP, Precision, Recall, BER và confusion matrix dùng để giải thích kết quả. Không tối ưu ngưỡng trên tập test.

Hướng mở rộng tiếp theo là giữ preprocessing và ba mô hình của 03c, áp dụng cùng tiêu chí chi phí và nested CV. Thử nghiệm ba mô hình theo mục tiêu mới **chưa chạy**.

## Cấu hình và ngưỡng cuối đã chọn trong 03b

- Dòng kết quả: `scenario = emc_r10`, `selection_scope = cost_selected`.
- Mô hình: Random Forest, `class_weight="balanced"`, 200 cây, `max_depth=None`, `min_samples_leaf=5`; pipeline chọn 80 đặc trưng bằng SelectKBest sau bốn bước preprocessing cũ.
- Candidate ID: `rf_k80_dNone_leaf5`.
- Ngưỡng: **0,344288**. Dùng đúng pipeline này: score Fail ≥ ngưỡng thì loại, nhỏ hơn ngưỡng thì cho qua.
- Điểm OOF dùng để lựa chọn: FP = 152, FN = 47, N = 1.253; EMC = `(152 + 10 × 47) / 1253` = **0,496409**.
- Điểm đánh giá quy trình bằng nested CV: FP = 246, FN = 45, N = 1.253; EMC = `(246 + 10 × 45) / 1253` = **0,555467**.

Khi viết báo cáo chất lượng, dùng **0,555467 từ nested CV**; điểm 0,496409 là điểm đã được dùng để lựa chọn, có thể lạc quan. Nested CV đánh giá quy trình lựa chọn, mỗi fold có thể có cấu hình/ngưỡng khác nhau; không phải mọi dự đoán vòng ngoài đều dùng ngưỡng cuối 0,344288.

Xem [bản tổng hợp dành cho người viết báo cáo](report/ket_luan_chi_phi_emc_r10.md) và [hướng dẫn notebook 03b](HUONG_DAN_03B_COST_SENSITIVE.md).

## Liên hệ với kết quả đã chạy

- `emc_r10` trong 03b đã dùng đúng bảng chi phí này.
- `review_r10` trong 03b và mục tiêu lựa chọn của 03c dùng TP = 1, nên thuộc giả định quy trình trước đó.
- Cột `emc_error_only` trong kết quả 03c tính được chi phí mới tại các ngưỡng cũ. Những ngưỡng/cấu hình này chưa được chọn để tối ưu mục tiêu mới, nên không thể coi đó là kết quả của lần tối ưu mới.
- Phần tìm ngưỡng trong 03b đã được viết lại bằng vòng lặp cơ bản. Mục 10.1 có ví dụ OOF thật và bảng 10 ngưỡng có chi phí thấp nhất. Đã chạy lại toàn bộ 03b bằng Python 3.11; bảng nested CV và policy cuối khớp kết quả đã có.
