import os
import matplotlib.pyplot as plt
import numpy as np

# 1. Định nghĩa đường dẫn gốc của dataset
# Thay đổi đường dẫn này cho phù hợp với máy của bạn
dataset_path = "Dataset/Data"  # Giả sử thư mục Dataset/Data nằm trong thư mục hiện tại

# Các tập con (splits) của dataset
splits = ["train", "valid", "test"]

# Định nghĩa rõ ràng 4 nhãn cấp cao mà bạn muốn hiển thị trên biểu đồ
target_labels = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma",
]

# Dictionary để lưu trữ số lượng mẫu cho từng nhãn mục tiêu trong từng tập
# Khởi tạo với số lượng 0 cho tất cả các nhãn mục tiêu trong mỗi split
label_counts = {split: {label: 0 for label in target_labels} for split in splits}


# 2. Ánh xạ các tên folder chi tiết về nhãn cấp cao
def map_label_to_target(label_name, target_labels):
    """Maps a detailed folder name to one of the target high-level labels."""
    label_name_lower = label_name.lower()

    # Kiểm tra các nhãn cấp cao trước
    if "normal" in label_name_lower and "normal" in target_labels:
        return "normal"
    elif "adenocarcinoma" in label_name_lower and "adenocarcinoma" in target_labels:
        return "adenocarcinoma"
    elif (
        "large.cell.carcinoma" in label_name_lower
        and "large.cell.carcinoma" in target_labels
    ):
        return "large.cell.carcinoma"
    elif (
        "squamous.cell.carcinoma" in label_name_lower
        and "squamous.cell.carcinoma" in target_labels
    ):
        return "squamous.cell.carcinoma"
    else:
        # Trường hợp không khớp với nhãn nào, có thể in cảnh báo hoặc bỏ qua
        # print(f"Cảnh báo: Tên folder '{label_name}' không khớp với nhãn mục tiêu nào.")
        return None  # Trả về None nếu không khớp


# 3. Thu thập số lượng mẫu, áp dụng ánh xạ
print(f"Đang quét dataset tại: {dataset_path}")
for split in splits:
    split_path = os.path.join(dataset_path, split)
    print(f"Kiểm tra thư mục: {split_path}")

    if not os.path.isdir(split_path):
        print(
            f"Cảnh báo: Thư mục '{split_path}' không tồn tại hoặc không phải là thư mục."
        )
        continue

    # Liệt kê các thư mục con bên trong split_path (đây là các nhãn thực tế trong folder)
    actual_labels_in_split = [
        d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))
    ]

    if not actual_labels_in_split:
        print(f"Cảnh báo: Không tìm thấy thư mục nhãn nào trong '{split_path}'.")
        continue

    print(
        f"Tìm thấy các thư mục nhãn thực tế trong '{split}': {actual_labels_in_split}"
    )

    for actual_label in actual_labels_in_split:
        mapped_label = map_label_to_target(actual_label, target_labels)

        # Chỉ xử lý nếu ánh xạ thành công về nhãn mục tiêu VÀ nhãn mục tiêu này có trong danh sách
        if mapped_label in target_labels:
            label_path = os.path.join(split_path, actual_label)
            # Đếm số lượng file trong thư mục nhãn thực tế
            count = len(
                [
                    name
                    for name in os.listdir(label_path)
                    if os.path.isfile(os.path.join(label_path, name))
                ]
            )

            # Cộng số lượng vào nhãn đã được ánh xạ
            label_counts[split][mapped_label] += count
        elif mapped_label is None:
            # Cảnh báo này có thể gây nhiễu nếu có nhiều folder không liên quan
            # print(f"Cảnh báo: Bỏ qua thư mục '{actual_label}' trong '{split}' vì không thể ánh xạ tới nhãn mục tiêu.")
            pass  # Bỏ qua các thư mục không phải là nhãn mục tiêu

# 4. Tính toán số lượng mẫu cho toàn bộ dataset (tổng của train, valid, test)
overall_counts = {label: 0 for label in target_labels}
for label in target_labels:
    overall_counts[label] = sum(label_counts[split].get(label, 0) for split in splits)

# 5. Chuẩn bị dữ liệu cho việc vẽ biểu đồ
# Lấy danh sách counts theo thứ tự của target_labels
overall_values = [overall_counts.get(label, 0) for label in target_labels]
test_values = [label_counts["test"].get(label, 0) for label in target_labels]
train_values = [label_counts["train"].get(label, 0) for label in target_labels]
valid_values = [label_counts["valid"].get(label, 0) for label in target_labels]

# 6. Vẽ biểu đồ 4 plot con
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Tạo figure và 4 axes (2x2 grid)
axes = axes.flatten()  # Làm phẳng mảng axes để dễ dàng truy cập bằng index

# Tiêu đề cho các plot
titles = ["Tổng cộng toàn bộ dataset", "Tập Test", "Tập Train", "Tập Valid"]
data_values = [overall_values, test_values, train_values, valid_values]

# Kiểm tra xem có dữ liệu để vẽ không (ít nhất là có nhãn mục tiêu)
if not target_labels:
    print("Không có nhãn mục tiêu nào được định nghĩa để vẽ biểu đồ.")
else:
    # Vẽ từng plot
    for i in range(4):
        ax = axes[i]
        counts = data_values[i]

        # Kiểm tra nếu tất cả counts đều là 0 cho plot này
        if not any(counts):
            ax.set_title(f"{titles[i]} (Không có dữ liệu)")
            ax.text(
                0.5,
                0.5,
                "Không có dữ liệu",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])  # Ẩn các tick nếu không có dữ liệu
            ax.set_yticks([])
        else:
            # Sử dụng target_labels làm nhãn trên trục x
            bars = ax.bar(target_labels, counts)
            ax.set_title(titles[i])
            ax.set_ylabel("Số lượng mẫu")

            # SỬA LỖI TẠI ĐÂY: Chỉ dùng rotation trong tick_params
            ax.tick_params(axis="x", rotation=45)  # <-- Chỉ giữ rotation

            # Thiết lập lại nhãn trục X với căn chỉnh (ha) sau khi đã có nhãn
            # Sử dụng target_labels vì chúng ta muốn hiển thị 4 nhãn này
            ax.set_xticklabels(
                target_labels, rotation=45, ha="right"
            )  # <-- Thêm dòng này để thiết lập lại nhãn với ha='right'

            # Hiển thị số lượng lên trên mỗi cột (tùy chọn)
            for bar in bars:
                yval = bar.get_height()
                if yval > 0:
                    # Căn chỉnh text cho phù hợp với cột xoay
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        yval,
                        int(yval),
                        va="bottom",
                        ha="center",
                        fontsize=8,
                    )  # va: verticalalignment, ha: horizontalalignment

    # Đặt tiêu đề chung cho toàn bộ figure
    fig.suptitle("Phân phối nhãn trong Dataset", fontsize=16)

    # Điều chỉnh layout để các plot không bị chồng lên nhau
    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )  # Điều chỉnh rect để chừa chỗ cho supertitle

    # Hiển thị biểu đồ
    plt.show()

    save_filename = "dataset_label_distribution.png"
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    print(f"\nBiểu đồ đã được lưu thành file: {save_filename}")

print("\nHoàn thành.")
