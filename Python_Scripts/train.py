from sklearn.model_selection import train_test_split, RandomizedSearchCV,StratifiedKFold
import pandas as pd
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import joblib
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path
import re
import shutil



param_grid_hgb = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_iter": [100, 200],
    "max_depth": [5, 10],
    "min_samples_leaf": [20, 50],
    "l2_regularization": [0, 0.1, 1]
}

param_grid_xgb = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "scale_pos_weight": [5, 10, 20]
}

def load_data(path):
    df = pd.read_csv(path)
    return df

def data_splitting_and_resampling(df):
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2026, stratify = y)
    sm = SMOTE(random_state = 2026)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    return X_train_res, y_train_res, X_test, y_test

def create_model(model_name):
    if model_name == 'xgboost':
        model = XGBClassifier(random_state=2026, use_label_encoder=False, eval_metric='logloss')
    elif model_name == 'hgb':
        model = HistGradientBoostingClassifier(random_state=2026)
    return model

def train_model_sweep(X_train_res, y_train_res, X_test, y_test, model):
    search_best_model = RandomizedSearchCV(model, param_grid_xgb, n_iter=10, cv=StratifiedKFold(n_splits=5), scoring='recall', random_state=2026, n_jobs=-1, verbose=1)
    search_best_model.fit(X_train_res, y_train_res)
    best_model = search_best_model.best_estimator_
    return best_model

def evaluste_model(model, X_train_res, y_train_res, X_test, y_test):
    model_name = type(model).__name__
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1_score = report['1']['f1-score']
    accuracy = accuracy_score(y_test, y_pred)
    model_performance = {
        'model_name': model_name,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }

    return model_performance

def select_best_model(model_performance_list):
    best_model = max(model_performance_list, key=lambda x: x['f1_score'])
    return best_model


def replace_model(model, model_name):
    folder = Path("../App")
    pkl_file = next(folder.glob("*.pkl"))
    version = re.search(r'v(\d+)', pkl_file.name).group(1)
    version = int(version) + 1
    new_model_name = f'best_model_v{version}.pkl'
    joblib.dump(model, folder / new_model_name)
    old_model_path = folder / pkl_file.name
    if old_model_path.exists():
        old_model_path.unlink()
        destination_folder = "../Models/archive"
        shutil.move(old_model_path, destination_folder/pkl_file.name)



def train_and_save_model():
    df = load_data('..\Data Files\Processed Files\preprocessed_data.csv')
    X_train_res, y_train_res, X_test, y_test = data_splitting_and_resampling(df)
    models = ['xgboost', 'hgb']
    model_performance_list = []
    for model in models:
        model = create_model(model)
        best_model = train_model_sweep(X_train_res, y_train_res, X_test, y_test, model)
        model_performance = evaluste_model(best_model, X_train_res, y_train_res, X_test, y_test)
        model_performance_list.append(model_performance)
    best_model = select_best_model(model_performance_list)
    replace_model(best_model, model_performance['model_name'])

if __name__ == "__main__":
    train_and_save_model()


    
