import os
import json
from tqdm import tqdm

from models.models import *
from custom_dataset import *

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost multi-class trainer")
    
    # Boolean flags
    parser.add_argument('--do_train', action='store_true', help='Thực hiện training nếu có cờ này')
    parser.add_argument('--do_predict', action='store_true', help='Thực hiện predict nếu có cờ này')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    data_directory = os.path.join("Dataset", "Data")
    save_checkpoint_path = "Checkpoints"
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

    feature_extractors = {
        "hog": HOGFeatureExtractor(),
        "lbp": LBPFeatureExtractor(),
        "htfs": GLCMFeatureExtractor(),
        "gabor": GaborExtractor(),
        "sift": SIFTFeatureExtractor()
    }

    models = {}

    # =========INIT MODELS==========
    # 1. SVM
    for name, extractor in feature_extractors.items():
        model = SVMClassifier(
            C=1, 
            kernel='rbf', 
            degree=3, 
            random_state=42,
            feature_extractor=extractor,
            model_path=os.path.join(save_checkpoint_path, f"svm_{name}_model.pkl")
        )
        models[f"svm_{name}"] = model

    # 2. Logistic Regression
    for name, extractor in feature_extractors.items():
        model = LRClassifier(
            C=1.0, 
            solver='lbfgs', 
            max_iter=1000, 
            random_state=42,
            multi_class='multinomial',
            feature_extractor=extractor,
            model_path=os.path.join(save_checkpoint_path, f"lr_{name}_model.pkl")
        )
        models[f"lr_{name}"] = model

    # 3. Random forest
    for name, extractor in feature_extractors.items():
        model = RFClassifier(
            n_estimators=100, 
            criterion='gini', 
            random_state=42,
            feature_extractor=extractor,
            model_path=os.path.join(save_checkpoint_path, f"rf_{name}_model.pkl")
        )
        models[f"rf_{name}"] = model

    # 4. XGBoost
    for name, extractor in feature_extractors.items():
        model = XGBoostClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=6,
            random_state=42,
            objective='multi:softmax', 
            num_class=4,
            feature_extractor=extractor,
            model_path=os.path.join(save_checkpoint_path, f"xgb_{name}_model.pkl")
        )
        models[f"xgb_{name}"] = model

    # =========TRAIN MODELS==========
    if args.do_train:
        print("*"*20+"[TRAINING]"+"*"*20)
        for name, model in tqdm(models.items()):
            model.train(train_data)

    # =========EVALUATE MODELS==========
    if args.do_predict:
        result_path = os.path.join("Result", "ML_results.json")
        print("*"*20+"[EVALUATE]"+"*"*20)
        os.makedirs("Result", exist_ok=True)
        for name, model in tqdm(models.items()):
            y_pred, metrics = model.predict(test_data)

            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = {}

            data[name] = metrics
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
