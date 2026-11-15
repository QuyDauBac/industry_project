# Đối chiếu cách lựa chọn cấu hình và ngưỡng trong 03b, 03c

Cập nhật theo quyết định ngày 15/11/2026. Kết quả để viết báo cáo nằm trong [bản tổng hợp EMC](report/ket_luan_chi_phi_emc_r10.md).

## 1. Hướng đã chốt

Dùng **cách chọn cấu hình và ngưỡng theo chi phí của 03b**, với kịch bản **`emc_r10`**:

\[
EMC=(FP+10FN)/N.
\]

Quy trình mới là cho qua sản phẩm dự đoán Pass, loại ngay sản phẩm dự đoán Fail, không kiểm tra lại. TN = TP = 0, FP = 1, FN = 10 là các chi phí giả định. Xem [quy ước đã chốt](QUY_UOC_CHI_PHI_DA_CHOT.md).

## 2. Khác biệt giữa hai notebook đã chạy

| Nội dung | 03b | 03c |
|---|---|---|
| Mô hình | LR và RF có trọng số lớp | XGBoost, LR L1 và RF có trọng số lớp |
| Pipeline | Bốn bước preprocessing cũ, thêm SelectKBest | Bốn bước preprocessing từ 02, không SelectKBest |
| Chọn cấu hình trong từng họ | Chi phí OOF thấp nhất sau khi tối ưu ngưỡng | OOF AP cao nhất; chi phí dùng để phá hòa |
| Chọn ngưỡng mỗi cấu hình | Chi phí OOF thấp nhất theo kịch bản | Chi phí OOF thấp nhất theo công thức cũ `(TP+FP+10FN)/N` |
| Chọn họ mô hình | `cost_selected` chọn LR/RF trong vòng trong | Chi phí OOF trung bình qua seed, chốt trước test |
| Đánh giá | Nested CV: outer 5, inner 3, trên train ban đầu | Năm seed CV/mô hình, cùng một holdout 314 mẫu |

03c có tập test tách khỏi bước lựa chọn; thiết kế đó hợp lệ. Nested CV giúp đánh giá toàn bộ quy trình chọn cấu hình và ngưỡng qua nhiều lượt chia. Nó tăng số lượt train, không tự làm mô hình tốt hơn.

## 3. Hai cách quét ngưỡng có tương đương không?

Với **cùng score, cùng bảng chi phí và cùng quy tắc phá hòa**, hai notebook thử cùng các điểm cắt làm thay đổi nhãn dự đoán. Chúng đều thêm phương án gán Fail tất cả và cho qua tất cả. Quy tắc là `score >= threshold` thì Fail.

Đã đối chiếu trên 15 bộ score OOF lưu từ 03c, với bảng chi phí cũ TN=0, FP=1, TP=1, FN=10: **15/15 bộ trả về cùng ngưỡng** khi dùng hai hàm. Vì vậy, khác biệt chính là tiêu chí chọn cấu hình và thiết kế đánh giá; việc lấy điểm cắt từ đường PR không tạo ra mục tiêu chi phí mới.

Phần tìm ngưỡng trong 03b hiện được viết bằng vòng lặp `for`, đếm số nhầm, tính chi phí rồi sắp xếp. Mục 10.1 minh họa trên OOF thật với công thức EMC đã chốt. Kiểm tra hồi quy cho thấy hàm basic và hàm trước đó cho cùng ngưỡng/các chỉ số trên 90 tổ hợp OOF × kịch bản chi phí.

## 4. Vì sao ưu tiên chi phí thay cho AP khi chọn cấu hình?

AP tổng hợp Precision trên đường Precision–Recall qua nhiều ngưỡng, không sử dụng tỷ lệ thiệt hại FP/FN của doanh nghiệp. Cấu hình AP cao nhất có thể có chi phí cao hơn tại vùng ngưỡng vận hành.

Minh họa từ **thử nghiệm cũ của 03c, dùng TP=1**: với LR L1 seed 13, C=0,1 có AP 0,130701 và chi phí 0,592977; C=0,01 có AP 0,127271 nhưng chi phí thấp hơn, 0,537909. Trên bảng OOF cũ, 7/15 nhóm seed × họ mô hình có cấu hình cho chi phí thấp hơn cấu hình AP chọn.

Các số này chỉ minh họa sự khác nhau giữa mục tiêu AP và chi phí trong thử nghiệm cũ. Chúng chưa phải kết quả tối ưu mới theo EMC hiện tại và không chứng minh mức cải thiện trên test.

## 5. Tại sao cần tách điểm lựa chọn và điểm đánh giá?

OOF bảo đảm từng score được tạo bởi mô hình chưa học mẫu đó. Nhưng nếu dùng nhãn của cả bộ OOF để chọn cấu hình/ngưỡng tốt nhất, điểm của lựa chọn thắng cuộc trên chính bộ OOF có thể lạc quan.

Trong nested CV, vòng trong tạo OOF và chọn policy; fold ngoài chưa tham gia lựa chọn được dùng để đánh giá policy đó. Dự đoán vòng ngoài cũng là OOF, nhưng độc lập với bước lựa chọn tương ứng. Năm seed trên cùng dữ liệu không tự tạo ra năm tập test độc lập.

Ở 03b hiện tại:

- Điểm đánh giá `emc_r10/cost_selected`: **EMC = 0,555467** từ nested CV.
- Policy cuối: RF `rf_k80_dNone_leaf5`, ngưỡng **0,344288**; điểm OOF dùng để chọn là **0,496409**.
- Mỗi fold ngoài có policy riêng. Không gắn ngưỡng cuối 0,344288 cho toàn bộ số đo nested CV.

## 6. Việc đã làm và hướng mở rộng

Đã cập nhật 03b sang trình bày `emc_r10` làm kịch bản chính, đơn giản hóa hàm tìm ngưỡng, thêm ví dụ OOF thật và chạy lại toàn bộ bằng Python 3.11. Bảng nested CV và policy cuối khớp số đo đã có ở lần chạy trước.

Hướng mở rộng là dùng ba mô hình và preprocessing của 03c, chọn cấu hình/ngưỡng theo EMC mới, rồi đánh giá nested CV. **Thử nghiệm mở rộng này chưa chạy.** Không đổi tên các số đo 03c cũ thành kết quả của phương pháp mới và không so trực tiếp chi phí 03b/03c để quy kết cải thiện, vì khác bảng chi phí, pipeline và tập đánh giá.

## Tham khảo

- [scikit-learn: lựa chọn ngưỡng](https://scikit-learn.org/stable/modules/classification_threshold.html).
- [scikit-learn: cost-sensitive learning](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html).
- [scikit-learn: nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
- Mã và bảng kết quả của dự án được dẫn trong [bản tổng hợp EMC](report/ket_luan_chi_phi_emc_r10.md).
