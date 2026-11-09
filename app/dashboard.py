"""Dashboard hỗ trợ quyết định cho dự đoán lỗi sản xuất bán dẫn (SECOM).

Trạng thái: BỘ KHUNG, chưa có code giao diện.
Dự kiến dùng lại pipeline, model, ngưỡng trong `artifacts/` và các hàm đã thiết kế
ở `05_deployment_demo.ipynb`.
"""


# ---------------------------------------------------------------- Khu vực 1: Trạng thái hiện tại
# TODO: hiển thị lô đang xét, thời điểm, số lô đã xử lý.

# ---------------------------------------------------------------- Khu vực 2: Risk score
# TODO: hiển thị xác suất Fail của lô và so với ngưỡng quyết định.

# ---------------------------------------------------------------- Khu vực 3: Cảnh báo
# TODO: hiển thị cảnh báo khi risk score vượt ngưỡng; danh sách cảnh báo gần đây.

# ---------------------------------------------------------------- Khu vực 4: Khuyến nghị hành động
# TODO: hiển thị khuyến nghị (ví dụ kiểm tra thêm, giữ lô, cho qua) theo mức rủi ro.

# ---------------------------------------------------------------- Khu vực 5: Nút kỹ sư xác nhận
# TODO: nút xác nhận / bác bỏ cảnh báo của kỹ sư (human-in-the-loop) và ghi lại phản hồi.


def main() -> None:
    """Điểm vào của dashboard."""
    # TODO
    pass


if __name__ == "__main__":
    main()
