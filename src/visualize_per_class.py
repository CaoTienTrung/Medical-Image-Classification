import os
import matplotlib.pyplot as plt
import numpy as np

# 1. Định nghĩa đường dẫn gốc của dataset
# Thay đổi đường dẫn này cho phù hợp với máy của bạn
dataset_path = "Dataset/Data"  # Giả sử thư mục Dataset/Data nằm trong thư mục hiện tại

# Các tập con (splits) của dataset
splits = ["train", "valid", "test"]

# Dictionary để lưu trữ số lượng mẫu cho từng nhãn trong từng tập
label_counts = {split: {} for split in splits}

# Danh sách tất cả các nhãn dự kiến (dựa trên ảnh test)
# Ta sẽ khám phá động các nhãn có trong dataset
all_labels = set()

# 2. Thu thập số lượng mẫu cho từng nhãn trong từng tập
print(f"Đang quét dataset tại: {dataset_path}")
for split in splits:
    split_path = os.path.join(dataset_path, split)
    print(f"Kiểm tra thư mục: {split_path}")

    if not os.path.isdir(split_path):
        print(
            f"Cảnh báo: Thư mục '{split_path}' không tồn tại hoặc không phải là thư mục."
        )
        continue  # Bỏ qua nếu thư mục split không tồn tại

    # Liệt kê các thư mục con bên trong split_path (đây chính là các nhãn)
    labels_in_split = [
        d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))
    ]

    if not labels_in_split:
        print(f"Cảnh báo: Không tìm thấy thư mục nhãn nào trong '{split_path}'.")
        continue

    print(f"Tìm thấy các nhãn trong '{split}': {labels_in_split}")

    for label in labels_in_split:
        label_path = os.path.join(split_path, label)
        # Đếm số lượng file trong thư mục nhãn
        count = len(
            [
                name
                for name in os.listdir(label_path)
                if os.path.isfile(os.path.join(label_path, name))
            ]
        )

        label_counts[split][label] = count
        all_labels.add(label)  # Thêm nhãn vào danh sách tổng

# Chuyển set nhãn tổng thể thành list và sắp xếp cho đồng nhất
all_labels = sorted(list(all_labels))

# 3. Tính toán số lượng mẫu cho toàn bộ dataset (tổng của train, valid, test)
overall_counts = {}
for label in all_labels:
    overall_counts[label] = sum(
        label_counts[split].get(label, 0) for split in splits
    )  # .get(label, 0) để tránh lỗi nếu nhãn không có trong 1 split nào đó

# 4. Chuẩn bị dữ liệu cho việc vẽ biểu đồ
# Lấy danh sách counts theo thứ tự của all_labels
overall_values = [overall_counts.get(label, 0) for label in all_labels]
test_values = [label_counts["test"].get(label, 0) for label in all_labels]
train_values = [label_counts["train"].get(label, 0) for label in all_labels]
valid_values = [label_counts["valid"].get(label, 0) for label in all_labels]

# 5. Vẽ biểu đồ 4 plot con
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Tạo figure và 4 axes (2x2 grid)
axes = axes.flatten()  # Làm phẳng mảng axes để dễ dàng truy cập bằng index

# Tiêu đề cho các plot
titles = ["Tổng cộng toàn bộ dataset", "Tập Test", "Tập Train", "Tập Valid"]
data_values = [overall_values, test_values, train_values, valid_values]

# Kiểm tra xem có dữ liệu để vẽ không
if not all_labels:
    print("Không tìm thấy bất kỳ nhãn nào trong dataset để vẽ biểu đồ.")
else:
    # Vẽ từng plot
    for i in range(4):
        ax = axes[i]
        counts = data_values[i]

        if not any(counts):  # Kiểm tra nếu tất cả counts đều là 0
            ax.set_title(f"{titles[i]} (Không có dữ liệu)")
            ax.text(
                0.5,
                0.5,
                "Không có dữ liệu",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            bars = ax.bar(all_labels, counts)
            ax.set_title(titles[i])
            ax.set_ylabel("Số lượng mẫu")
            ax.tick_params(
                axis="x", rotation=45, ha="right"
            )  # Xoay nhãn trục x để dễ đọc

            # Hiển thị số lượng lên trên mỗi cột (tùy chọn)
            for bar in bars:
                yval = bar.get_height()
                if yval > 0:
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

print("\nHoàn thành.")
