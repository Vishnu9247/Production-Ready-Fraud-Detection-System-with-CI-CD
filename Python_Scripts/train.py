from pathlib import Path
import re
import warnings

import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

warnings.filterwarnings("ignore")


PARAM_GRID_XGB = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "scale_pos_weight": [5, 10, 20],
}

PARAM_GRID_HGB = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_iter": [100, 200],
    "max_depth": [5, 10],
    "min_samples_leaf": [20, 50],
    "l2_regularization": [0, 0.1, 1],
}


def load_data(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")

    return pd.read_csv(path)


def split_and_resample_data(df):
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=2026,
        stratify=y,
    )

    smote = SMOTE(random_state=2026)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, y_train_res, X_test, y_test


def create_model(model_name):
    if model_name == "xgboost":
        return XGBClassifier(
            random_state=2026,
            eval_metric="logloss",
        )

    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            random_state=2026,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def get_param_grid(model_name):
    if model_name == "xgboost":
        return PARAM_GRID_XGB

    if model_name == "hgb":
        return PARAM_GRID_HGB

    raise ValueError(f"Unknown model name: {model_name}")


def tune_model(model, model_name, X_train_res, y_train_res):
    param_grid = get_param_grid(model_name)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=2026,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        scoring="recall",
        cv=cv,
        random_state=2026,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train_res, y_train_res)

    return search.best_estimator_, search.best_params_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return {
        "model_name": type(model).__name__,
        "model": model,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "accuracy": accuracy_score(y_test, y_pred),
    }


def select_best_model(model_results):
    return max(model_results, key=lambda x: x["recall"])


def get_next_model_version(app_folder):
    existing_models = list(app_folder.glob("*.pkl"))

    if not existing_models:
        return 1, None

    old_model_path = existing_models[0]

    match = re.search(r"v(\d+)", old_model_path.name)

    if match:
        next_version = int(match.group(1)) + 1
    else:
        next_version = 1

    return next_version, old_model_path


def archive_old_model(old_model_path, archive_folder):
    if old_model_path is None:
        return

    archive_folder.mkdir(parents=True, exist_ok=True)

    archived_model_path = archive_folder / old_model_path.name

    old_model_path.rename(archived_model_path)

    print(f"Archived old model: {archived_model_path}")


def save_new_model(model, app_folder, version):
    app_folder.mkdir(parents=True, exist_ok=True)

    new_model_name = f"best_model_v{version}.pkl"
    new_model_path = app_folder / new_model_name

    joblib.dump(model, new_model_path)

    print(f"Saved new model: {new_model_path}")

    return new_model_path


def replace_model(model):
    app_folder = Path("App")
    archive_folder = Path("Models") / "archive"

    next_version, old_model_path = get_next_model_version(app_folder)

    archive_old_model(old_model_path, archive_folder)

    new_model_path = save_new_model(model, app_folder, next_version)

    return new_model_path


def train_and_save_model():
    data_path = Path("Data Files") / "Processed Files" / "preprocessed_data.csv"

    df = load_data(data_path)

    X_train_res, y_train_res, X_test, y_test = split_and_resample_data(df)

    model_names = ["xgboost", "hgb"]

    model_results = []

    for model_name in model_names:
        print(f"\nTraining model: {model_name}")

        model = create_model(model_name)

        best_model, best_params = tune_model(
            model=model,
            model_name=model_name,
            X_train_res=X_train_res,
            y_train_res=y_train_res,
        )

        result = evaluate_model(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
        )

        result["best_params"] = best_params

        model_results.append(result)

        print(f"Model: {result['model_name']}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"F1 Score: {result['f1_score']:.4f}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Best Params: {best_params}")

    best_result = select_best_model(model_results)

    print("\nBest model selected:")
    print(f"Model: {best_result['model_name']}")
    print(f"Recall: {best_result['recall']:.4f}")
    print(f"F1 Score: {best_result['f1_score']:.4f}")

    saved_model_path = replace_model(best_result["model"])

    print(f"\nFinal model saved at: {saved_model_path}")


if __name__ == "__main__":
    train_and_save_model()