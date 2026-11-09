"""Hàm tiện ích dùng chung cho đồ án SECOM (Ứng dụng AI trong công nghiệp).

Module này gom các hằng số và hàm dùng lại ở nhiều notebook để mọi notebook
dùng cùng một cách nạp dữ liệu, chia train/test, đánh giá và lưu/nạp artifact.

Trạng thái: BỘ KHUNG. Các hàm chưa được cài đặt (raise NotImplementedError).
Không chứa logic xử lý dữ liệu.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

# ---------------------------------------------------------------- Hằng số
SEED: int = 42
TEST_SIZE: float = 0.20
MISSING_THRESHOLD: float = 0.70

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data" / "secom"
ARTIFACT_DIR: Path = PROJECT_ROOT / "artifacts"
RESULTS_DIR: Path = ARTIFACT_DIR / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"


# ---------------------------------------------------------------- Hàm
def load_secom(data_dir: Path = DATA_DIR) -> tuple["pd.DataFrame", "pd.Series", "pd.Series"]:
    """Đọc dữ liệu SECOM từ thư mục `data_dir`.

    Đầu vào:
        data_dir: thư mục chứa `secom.data` và `secom_labels.data`.

    Đầu ra:
        Bộ ba (X, y, timestamp):
            X: DataFrame 590 cột đặc trưng (giữ NaN).
            y: Series nhãn đã đổi sang 0 (Pass) / 1 (Fail).
            timestamp: Series thời gian của từng lô, cùng index với X.
    """
    import pandas as pd

    data_dir = Path(data_dir)

    # Đọc ma trận đặc trưng: phân cách bằng khoảng trắng, không có header
    X = pd.read_csv(data_dir / "secom.data", sep=r"\s+", header=None)
    X = X.astype(float)

    # Đọc nhãn bằng engine mặc định: cột 0 là nhãn, cột 1 là ngày giờ (trong dấu ngoặc kép)
    labels_raw = pd.read_csv(data_dir / "secom_labels.data", sep=r"\s+", header=None)
    if labels_raw.shape[1] == 3:
        # Phòng trường hợp ngày và giờ bị tách thành hai cột
        time_text = labels_raw[1].astype(str) + " " + labels_raw[2].astype(str)
    else:
        time_text = labels_raw[1].astype(str)

    # Đổi nhãn: -1 -> 0 (Pass), 1 -> 1 (Fail)
    y = labels_raw[0].map({-1: 0, 1: 1}).astype(int)
    y.name = "label"

    # Parse timestamp theo định dạng ngày/tháng/năm giờ:phút:giây
    timestamp = pd.to_datetime(time_text, format="%d/%m/%Y %H:%M:%S")
    timestamp.name = "timestamp"

    return X, y, timestamp


def get_split(
    y: "pd.Series",
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Chia chỉ số train/test theo kiểu stratified.

    Đầu vào:
        y: nhãn 0/1 dùng để phân tầng.
        test_size: tỉ lệ test (mặc định 0.20).
        seed: random seed (mặc định 42).

    Đầu ra:
        Cặp (train_idx, test_idx): mảng chỉ số vị trí của hai tập.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Chia trên mảng chỉ số vị trí, phân tầng theo y (giống Lab 02: 80/20, random_state=42)
    all_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=seed, stratify=y
    )
    return train_idx, test_idx


def evaluate_classifier(
    y_true: "np.ndarray",
    y_score: "np.ndarray",
    threshold: float = 0.5,
) -> dict[str, float]:
    """Tính các chỉ số đánh giá cho bài toán phân loại Fail.

    Đầu vào:
        y_true: nhãn thật 0/1.
        y_score: điểm/xác suất lớp Fail.
        threshold: ngưỡng quyết định; dự đoán Fail khi `y_score >= threshold`.

    Đầu ra:
        dict tên chỉ số -> giá trị: threshold, accuracy, precision, recall, f1 (của Fail),
        roc_auc, average_precision, balanced_accuracy, ber, tn, fp, fn, tp, fpr, fnr,
        alarm_rate. Chỉ số không xác định (mẫu số bằng 0) trả NaN; Precision/F1 dùng
        quy ước zero_division = 0.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    # Đổi điểm thành nhãn bằng ngưỡng, rồi đếm ma trận nhầm lẫn
    y_pred = (y_score >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    n_total = tn + fp + fn + tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    balanced_accuracy = (recall + (1 - fpr)) / 2
    has_both_classes = len(np.unique(y_true)) == 2

    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / n_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both_classes else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_score)) if has_both_classes else float("nan"),
        "balanced_accuracy": balanced_accuracy,
        "ber": 1 - balanced_accuracy,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": fpr,
        "fnr": fnr,
        "alarm_rate": (tp + fp) / n_total,
    }


def confusion_counts(y_true: "np.ndarray", y_pred: "np.ndarray") -> dict[str, int]:
    """Đếm số tuyệt đối của ma trận nhầm lẫn.

    Đầu vào:
        y_true: nhãn thật 0/1.
        y_pred: nhãn dự đoán 0/1.

    Đầu ra:
        dict với các khoá "tn", "fp", "fn", "tp" (kiểu int). Luôn dùng labels=[0, 1]
        nên ma trận luôn 2x2 kể cả khi một lớp không xuất hiện.
    """
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def bootstrap_ci(
    y_true: "np.ndarray",
    y_score: "np.ndarray",
    metric_fn: Callable[[Any, Any], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float]:
    """Khoảng tin cậy bootstrap cho một chỉ số.

    Đầu vào:
        y_true: nhãn thật 0/1.
        y_score: điểm/xác suất lớp Fail (hoặc nhãn dự đoán, tuỳ `metric_fn`).
        metric_fn: hàm nhận (y_true, y_score) và trả về một số thực.
        n_boot: số lần lấy mẫu bootstrap.
        alpha: mức ý nghĩa (0.05 -> khoảng tin cậy 95%).
        seed: random seed.

    Đầu ra:
        Cặp (cận dưới, cận trên) của khoảng tin cậy.
    """
    raise NotImplementedError


def choose_threshold(
    y_true: "np.ndarray",
    y_score: "np.ndarray",
    strategy: str = "f1",
) -> float:
    """Chọn ngưỡng quyết định.

    Chỉ gọi với dự đoán out-of-fold của tập train, không dùng test.

    Đầu vào:
        y_true: nhãn thật 0/1 của train.
        y_score: điểm/xác suất lớp Fail out-of-fold.
        strategy: hiện chỉ hỗ trợ "f1" (F1 của Fail cao nhất; F1 bằng nhau thì chọn
            ngưỡng cao hơn). Giá trị khác báo ValueError.

    Đầu ra:
        Ngưỡng được chọn (float), là một trong các giá trị điểm phân biệt của `y_score`;
        dự đoán Fail khi `y_score >= ngưỡng`.
    """
    import numpy as np

    if strategy != "f1":
        raise ValueError(f"strategy '{strategy}' chưa được hỗ trợ; chỉ có 'f1'.")

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    best_threshold = None
    best_f1 = -1.0
    # Duyệt ngưỡng từ thấp đến cao; dùng ">=" để khi F1 bằng nhau thì ngưỡng cao hơn thay thế
    for threshold in np.unique(y_score):
        y_pred = y_score >= threshold
        tp = int(((y_true == 1) & y_pred).sum())
        fp = int(((y_true == 0) & y_pred).sum())
        fn = int(((y_true == 1) & ~y_pred).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        if f1 >= best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def save_artifact(obj: Any, name: str, artifact_dir: Path = ARTIFACT_DIR) -> Path:
    """Lưu một đối tượng (pipeline, model, ngưỡng, chỉ số chia, ...) bằng joblib.

    Đầu vào:
        obj: đối tượng cần lưu.
        name: tên file (ví dụ "model_rf.joblib").
        artifact_dir: thư mục đích (mặc định `artifacts/`).

    Đầu ra:
        Đường dẫn file đã lưu.
    """
    import joblib

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / name
    joblib.dump(obj, path)
    return path


def load_artifact(name: str, artifact_dir: Path = ARTIFACT_DIR) -> Any:
    """Nạp lại một đối tượng đã lưu bằng `save_artifact`.

    Đầu vào:
        name: tên file trong `artifact_dir`.
        artifact_dir: thư mục chứa artifact (mặc định `artifacts/`).

    Đầu ra:
        Đối tượng đã nạp.
    """
    import joblib

    path = Path(artifact_dir) / name
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy artifact '{name}' trong {Path(artifact_dir)}. "
            "Hãy chạy notebook tạo ra nó trước (thứ tự 01 → 02 → 03 → 04)."
        )
    return joblib.load(path)
