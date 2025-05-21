import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import numpy as np
import math
import torch
from custom_dataset import *

from custom_dataset import *

def visualize_images_by_classname(dataset, class_name, num_images, save_path=None):
    if class_name not in dataset.class_names:
        raise ValueError(f"Class name {class_name} does not exist !!!")
    
    label_id = dataset.class_names.index(class_name)

    found = 0
    plt.figure(figsize=(25,10))
    for i in range(len(dataset)):
        img, label = dataset[i]
        if (isinstance(label, torch.Tensor)):
            if label.ndim > 0 and label.shape[0] > 1:
                label = torch.argmax(label).item()
            else: 
                label = int(label)

        if (label == label_id):
            img_np = img.numpy().transpose(1,2,0) #(Height, Width, CHannel)
            img_np = np.clip(img_np * 255, 0, 255).astype('uint8')

            plt.subplot(math.ceil(num_images/5), 5, found + 1)
            plt.imshow(img_np)
            plt.title(f"{class_name}")
            plt.axis('off')

            found +=1
            if found >= num_images:
                break
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"========== Saved at {save_path} =========")
    else:
        plt.show()

def visualize_label_distribution(dataset, save_path=None):
    colors = cm.CMRmap([i/len(dataset.class_names) for i in range(len(dataset.class_names))])

    labels = [label for _, label in dataset.samples]
    if dataset.labels == 'inferred':
        labels = [label for _, label in dataset.samples]

    label_dict = {}
    for i in range(len(dataset.class_names)):
        label_dict[i] = labels.count(i)
    
    unique_labels = list(label_dict.keys())
    counts = list(label_dict.values())

    plt.figure(figsize=(10, 6))
    plt.bar(unique_labels, counts, color=colors)
    plt.title('Biểu đồ phân phối các nhãn', fontweight ='bold', fontsize = 12, color = 'red')
    plt.xlabel('Nhãn', fontweight ='bold', fontsize = 12)
    plt.ylabel('Số lượng', fontweight ='bold', fontsize = 12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"========== Saved at {save_path} =========")
    else:
        plt.show()

if __name__ == "__main__":
    IMG_SIZE = 227
    IMG_CHANNEL = 3
    BATCH_SIZE = 256
    COLOR_MODE = "grayscale"
    DIRECTORY = "F:\Studies\Third_year\Computer_vision\Project\Dataset\Data"
    CLASS_NAMES = sorted(os.listdir(os.path.join(DIRECTORY, 'train')))
    dataset = CustomImageDataset(
        os.path.join(DIRECTORY, 'train'),
        labels='inferred',
        label_mode='int',
        class_names=CLASS_NAMES,
        color_mode=COLOR_MODE,
        image_size=(IMG_SIZE, IMG_SIZE),
        interpolation='bilinear'
    )

    for class_name in CLASS_NAMES:
        visualize_images_by_classname(
            dataset,
            class_name=class_name,
            num_images=5,
            save_path=f'F:\Studies\Third_year\Computer_vision\Project\Images\{class_name}.png'
        )

    visualize_label_distribution(
        dataset,
        save_path='F:\Studies\Third_year\Computer_vision\Project\Images\class_distribution.png'
    )
