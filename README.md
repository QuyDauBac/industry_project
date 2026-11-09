# Phân tích và dự đoán lỗi trong quá trình sản xuất bán dẫn (SECOM)

Đồ án cuối kỳ môn **Ứng dụng AI trong công nghiệp**.
Dataset SECOM: 1567 mẫu, 590 đặc trưng, nhãn Pass/Fail, có timestamp. Bài toán: classification.

Mỗi notebook ứng với một giai đoạn của lifecycle 8 bước (Problem Definition → Data Understanding → Data Preparation → Method Selection → Model Development → Evaluation → Deployment → Monitoring and Improvement). Notebook sau nạp lại kết quả notebook trước đã lưu trong `artifacts/`, không import notebook khác.

> Trạng thái: bộ khung. Các notebook và `secom_utils.py` mới có tiêu đề mục và `# TODO`.

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
