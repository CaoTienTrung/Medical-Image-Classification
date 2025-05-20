import os

from models.models import *
from custom_dataset import *

if __name__ == "__main__":
    data_directory = "F:\Studies\Third_year\Computer_vision\Project\ProjectCode\Dataset\Data"
    train_data = CustomImageDataset(
        directory=os.path.join(data_directory, 'train') ,
        label_mode='int',
        color_mode='grayscale',
        image_size=(224,224),
        interpolation='bilinear'
    )
    test_data = CustomImageDataset(
        directory=os.path.join(data_directory, 'test') ,
        label_mode='int',
        color_mode='grayscale',
        image_size=(224,224),
        interpolation='bilinear'
    )
    model = SVMClassifier()
    y_pred, metrics = model.predict(test_data)
    print(metrics)
