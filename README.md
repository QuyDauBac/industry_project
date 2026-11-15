# Phân tích và dự đoán lỗi trong quá trình sản xuất bán dẫn (SECOM)

Đồ án cuối kỳ môn **Ứng dụng AI trong công nghiệp**.
Dataset SECOM: 1567 mẫu, 590 đặc trưng, nhãn Pass/Fail, có timestamp. Bài toán: classification.

Mỗi notebook ứng với một giai đoạn của lifecycle 8 bước (Problem Definition → Data Understanding → Data Preparation → Method Selection → Model Development → Evaluation → Deployment → Monitoring and Improvement). Notebook sau nạp lại kết quả notebook trước đã lưu trong `artifacts/`, không import notebook khác.

## Quyết định và kết quả mới nhất — 15/11/2026

- **Dành cho người viết báo cáo:** [Hướng đi và kết quả chốt theo EMC](report/ket_luan_chi_phi_emc_r10.md).
- Quy trình: dự đoán Fail thì loại ngay; chi phí giả định **TN = TP = 0, FP = 1, FN = 10**; tối ưu `(FP + 10 × FN) / N`.
- [Notebook 03b](03b_cost_sensitive_modeling.ipynb), mục **10.1**, trình bày cách tìm ngưỡng bằng vòng lặp và OOF thật. Policy cuối là RF, ngưỡng **0,344288**; điểm đánh giá quy trình bằng nested CV là **0,555467**, khác điểm OOF dùng để lựa chọn.
- [Quy ước đã chốt](QUY_UOC_CHI_PHI_DA_CHOT.md) và [hướng dẫn chạy/đọc 03b](HUONG_DAN_03B_COST_SENSITIVE.md).
- [03c](03c_xgb_l1_rf_five_seeds.ipynb) giữ kết quả thử nghiệm trước. Thử nghiệm ba mô hình với tiêu chí EMC mới và nested CV **chưa chạy**.

Các nội dung cấu trúc và lifecycle bên dưới mô tả tổ chức chung của đồ án.

## Cấu trúc thư mục

```
industry4ai-secom/
├── data/secom/          # đặt secom.data và secom_labels.data (không đưa lên git)
├── artifacts/           # pipeline, model, ngưỡng do notebook lưu
│   └── results/         # bảng kết quả .csv
├── figures/             # hình chèn vào báo cáo
├── report/              # báo cáo
├── slides/              # slide
├── app/
│   └── dashboard.py     # dashboard hỗ trợ quyết định
├── 01_eda_3B.ipynb
├── 02_preprocessing.ipynb
├── 03_modeling.ipynb
├── 04_evaluation.ipynb
├── 05_deployment_demo.ipynb
├── 06_reliability_monitoring.ipynb
├── secom_utils.py
├── secom_preprocessing.py   # file của giảng viên, tự copy vào (xem bên dưới)
├── requirements.txt
├── README.md
└── .gitignore
```

## Notebook → lifecycle → bài lab → chương báo cáo

| Notebook | Bước lifecycle | Bài lab của thầy | Chương báo cáo |
|---|---|---|---|
| `01_eda_3B.ipynb` | 1. Problem Definition, 2. Data Understanding | Lab 01 | TODO (dự kiến chương 2–3) |
| `02_preprocessing.ipynb` | 3. Data Preparation | Lab 02 | TODO (dự kiến chương 4) |
| `03_modeling.ipynb` | 4. Method Selection, 5. Model Development | Lab 03, Lab 06 | TODO (dự kiến chương 5–6) |
| `04_evaluation.ipynb` | 6. Evaluation | Lab 03, Lab 06 | TODO (dự kiến chương 7) |
| `05_deployment_demo.ipynb` | 7. Deployment | Phần mới theo đề | TODO (dự kiến chương 8) |
| `06_reliability_monitoring.ipynb` | 8. Monitoring and Improvement | Phần mới theo đề | TODO (dự kiến chương 9) |

Cột "Chương báo cáo" cần đối chiếu lại với mục lục báo cáo thực tế.

## Thứ tự chạy

`01 → 02 → 03 → 04 → 05 → 06`

Chạy đúng thứ tự vì notebook sau đọc file do notebook trước lưu trong `artifacts/`. Dashboard (`app/dashboard.py`) chạy sau 05.

## Đặt dữ liệu và file của giảng viên

1. Đặt `secom.data` và `secom_labels.data` vào `data/secom/`.
2. Copy `secom_preprocessing.py` (file của giảng viên, chứa `MissingRateSelector`) vào thư mục gốc, cùng cấp với `secom_utils.py`. Không sửa file này.

## Tạo môi trường

Windows (PowerShell):

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Thành viên và phân công

<!-- TODO -->
