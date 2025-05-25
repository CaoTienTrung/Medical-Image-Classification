import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Dict ánh xạ tên chi tiết về tên class cha
CLASS_MAP = {
    "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib": "adenocarcinoma",
    "adenocarcinoma_left.lower.lobe_T1a_N0_M0_IA": "adenocarcinoma",
    "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa": "large.cell.carcinoma",
    "large.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": "large.cell.carcinoma",
    "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": "squamous.cell.carcinoma",
    # Thêm các ánh xạ khác nếu có
}

# Các class cha chuẩn hóa (giống test)
STANDARD_CLASSES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "squamous.cell.carcinoma",
    "normal",
]

# Tạo bảng màu cho các class
COLORS = plt.get_cmap("tab10").colors  # hoặc 'Set2', 'tab20'...
CLASS_COLORS = {cls: COLORS[i % len(COLORS)] for i, cls in enumerate(STANDARD_CLASSES)}


def get_class_distribution(base_dir="Dataset/Data"):
    """
    Gom số lượng ảnh về class cha theo dict ánh xạ cho train, valid; test giữ nguyên.
    """
    splits = ["train", "test", "valid"]
    distribution = defaultdict(lambda: defaultdict(int))

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
        classes = os.listdir(split_dir)
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                n_images = len(
                    [
                        f
                        for f in os.listdir(class_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                )
                # Chuẩn hóa tên class cho train/valid
                if split in ["train", "valid"]:
                    class_std = CLASS_MAP.get(class_name, class_name)
                else:
                    class_std = class_name
                distribution[split][class_std] += n_images
    return distribution


def plot_distribution(distribution, classes):
    splits = ["train", "test", "valid"]
    totals = {
        split: sum(distribution[split][cls] for cls in classes) for split in splits
    }
    total_per_class = {
        cls: sum(distribution[split][cls] for split in splits) for cls in classes
    }
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle("Phân bố số lượng ảnh theo lớp (chuẩn hóa)", fontsize=16)

    def plot_bar(ax, values, title):
        x = np.arange(len(classes))
        barlist = ax.bar(x, values, color=[CLASS_COLORS[cls] for cls in classes])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_ylabel("Số lượng ảnh")
        for i, v in enumerate(values):
            ax.text(i, v, str(v), ha="center", va="bottom")

    plot_bar(
        axes[0, 0], [total_per_class[cls] for cls in classes], "Tổng số ảnh theo lớp"
    )
    plot_bar(
        axes[0, 1],
        [distribution["train"][cls] for cls in classes],
        "Số lượng ảnh trong tập train",
    )
    plot_bar(
        axes[1, 0],
        [distribution["test"][cls] for cls in classes],
        "Số lượng ảnh trong tập test",
    )
    plot_bar(
        axes[1, 1],
        [distribution["valid"][cls] for cls in classes],
        "Số lượng ảnh trong tập valid",
    )
    # Thêm chú thích màu
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[cls]) for cls in classes]
    fig.legend(handles, classes, loc="upper right", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig("data_distribution_bars.png")
    plt.close()
    print("\nThông tin phân bố dữ liệu (chuẩn hóa):")
    print("-" * 50)
    print("\nTổng thể:")
    total_images = sum(total_per_class.values())
    for cls in classes:
        count = total_per_class[cls]
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        print(f"{cls}: {count} ảnh ({percentage:.1f}%)")
    for split in splits:
        print(f"\nTập {split}:")
        print(f"Tổng số ảnh: {totals[split]}")
        for cls in classes:
            count = distribution[split][cls]
            percentage = (count / totals[split]) * 100 if totals[split] > 0 else 0
            print(f"{cls}: {count} ảnh ({percentage:.1f}%)")


if __name__ == "__main__":
    distribution = get_class_distribution()
    plot_distribution(distribution, STANDARD_CLASSES)
