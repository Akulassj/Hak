import numpy as np
import json
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix
from scipy.stats import randint, uniform

app = Flask(__name__)

# Загрузка данных
def load_training_data(file_path="dataset_train.json"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[Прогресс] Загружено {len(data)} записей из {file_path}.")
        print(f"[Прогресс] Уникальные значения isCommercial: {set(record['isCommercial'] for record in data)}")
        return data
    except FileNotFoundError:
        print(f"[Ошибка] Файл {file_path} не найден.")
        raise
    except json.JSONDecodeError as e:
        print(f"[Ошибка] Ошибка декодирования JSON: {str(e)}.")
        raise

def load_test_data(file_path="dataset_test.json"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[Прогресс] Загружено {len(data)} записей из {file_path} для теста.")
        return data
    except FileNotFoundError:
        print(f"[Ошибка] Файл {file_path} не найден.")
        raise
    except json.JSONDecodeError as e:
        print(f"[Ошибка] Ошибка декодирования JSON: {str(e)}.")
        raise

# Вычисление среднего totalArea для buildingType
def compute_mean_total_area(data):
    df = pd.DataFrame(data)
    mean_areas = df.groupby('buildingType')['totalArea'].mean().to_dict()
    print(f"[Прогресс] Средние площади по типам зданий: {mean_areas}")
    return mean_areas

# Обработка выбросов
def cap_outliers(values, lower_percentile=25, upper_percentile=75, factor=1.5):
    q25, q75 = np.percentile(values, [lower_percentile, upper_percentile])
    iqr = q75 - q25
    lower_bound = q25 - factor * iqr
    upper_bound = q75 + factor * iqr
    capped_values = np.clip(values, lower_bound, upper_bound)
    return capped_values


def prepare_features_and_target(data, mean_areas, encoder=None, fit_encoder=False):
    X_numeric, X_categorical, y = [], [], []
    feature_names_numeric = [
        "mean_consumption_per_area",
        "mean_oct_to_apr_per_area",
        "median_consumption_per_area",
        "max_consumption_per_area",
        "min_consumption_per_area",
        "consumption_variation",
        "winter_consumption_ratio",
        "residents_per_area",
        "rooms_count",
        "residents_count",
        "high_consumption_flag"
    ]

    for record in data:
        consumption_values = list(record.get("consumption", {}).values())
        if not consumption_values:
            print(
                f"[Предупреждение] Пустое потребление для записи {record.get('accountId', 'unknown')}, запись пропущена.")
            continue

        # Обработка выбросов
        consumption_values = cap_outliers(np.array(consumption_values))


        total_area = float(record.get("totalArea", mean_areas.get(record.get("buildingType", "Частный"), 100.0)))
        total_area = max(total_area, 1.0)

        # Признаки потребления
        mean_consumption = np.mean(consumption_values) / total_area
        oct_to_apr = [v for k, v in record.get("consumption", {}).items() if int(k) in [1, 2, 3, 4, 10, 11, 12]]
        mean_oct_to_apr = np.mean(oct_to_apr) / total_area if oct_to_apr else mean_consumption
        median_consumption = np.median(consumption_values) / total_area
        max_consumption = np.max(consumption_values) / total_area
        min_consumption = np.min(consumption_values) / total_area

        # Новые признаки
        consumption_variation = np.std(consumption_values) / (np.mean(consumption_values) + 1e-6)
        winter_consumption_ratio = np.mean(oct_to_apr) / (np.mean(consumption_values) + 1e-6) if oct_to_apr else 1.0
        residents_per_area = record.get("residentsCount", 1) / total_area
        high_consumption_flag = 1 if any(v > 3000 for v in oct_to_apr) else 0

        features_numeric = [
            mean_consumption,
            mean_oct_to_apr,
            median_consumption,
            max_consumption,
            min_consumption,
            consumption_variation,
            winter_consumption_ratio,
            residents_per_area,
            record.get("roomsCount", 1),
            record.get("residentsCount", 1),
            high_consumption_flag
        ]
        X_numeric.append(features_numeric)
        X_categorical.append([record.get("buildingType", "Частный")])
        y.append(1 if record["isCommercial"] else 0)

    X_numeric = np.array(X_numeric)
    y = np.array(y)

    # Кодирование buildingType
    if fit_encoder:
        encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        X_categorical_encoded = encoder.fit_transform(X_categorical)
    else:
        X_categorical_encoded = encoder.transform(X_categorical)

    # Формирование имен признаков
    feature_names = list(encoder.get_feature_names_out(['buildingType'])) + feature_names_numeric
    print(f"[Прогресс] Подготовлено {len(X_numeric)} образцов с {len(feature_names)} признаками")
    return X_numeric, X_categorical_encoded, y, feature_names, encoder

# Инициализация моделей и данных
models = {}
scalers = {}
data_store = {}
feature_names = []
encoder = None

def load_or_train_models():
    global feature_names, encoder
    if not os.path.exists('models'):
        os.makedirs('models')

    print("[Прогресс] Загрузка данных для обучения...")
    training_data = load_training_data("dataset_train.json")
    mean_areas = compute_mean_total_area(training_data)
    X_numeric, X_categorical_encoded, y, feature_names, encoder = prepare_features_and_target(training_data, mean_areas,
                                                                                              fit_encoder=True)

    if len(X_numeric) == 0 or len(np.unique(y)) < 2:
        print("[Ошибка] Недостаточно данных для обучения или отсутствует вариативность меток.")
        raise ValueError("Недостаточно данных для обучения.")

    # Нормализация числовых признаков
    print("[Прогресс] Нормализация числовых признаков...")
    scaler = RobustScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    X = hstack([X_categorical_encoded, csr_matrix(X_numeric_scaled)])
    print(f"[Прогресс] Нормализация завершена.")

    # Учет несбалансированности классов
    class_counts = np.bincount(y)
    scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0
    print(f"[Прогресс] Соотношение классов (scale_pos_weight): {scale_pos_weight:.2f}")

    model_key = "general"
    model_files = [
        f"models/{model_key}_rf.joblib",
        f"models/{model_key}_xgb.joblib",
        f"models/{model_key}_scaler.joblib",
        f"models/{model_key}_encoder.joblib"
    ]

    if not all(os.path.exists(f) for f in model_files):
        print("[Прогресс] Обучение новых моделей...")

        # RandomForest
        print("[Прогресс] Обучение RandomForest...")
        rf_param_dist = {
            'n_estimators': randint(100, 300),
            'max_depth': [10, 20, None],
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 4)
        }
        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        rf_search = RandomizedSearchCV(rf, rf_param_dist, n_iter=20, cv=3, scoring='balanced_accuracy', n_jobs=-1,
                                       random_state=42)
        rf_search.fit(X, y)
        models[f"{model_key}_rf"] = rf_search.best_estimator_
        print(f"[Прогресс] Лучшие параметры RandomForest: {rf_search.best_params_}")

        # XGBoost
        print("[Прогресс] Обучение XGBoost...")
        xgb_param_dist = {
            'n_estimators': randint(100, 300),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.29),
            'subsample': uniform(0.7, 0.3)
        }
        xgb = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
        xgb_search = RandomizedSearchCV(xgb, xgb_param_dist, n_iter=20, cv=3, scoring='balanced_accuracy', n_jobs=-1,
                                        random_state=42)
        xgb_search.fit(X, y)
        models[f"{model_key}_xgb"] = xgb_search.best_estimator_
        print(f"[Прогресс] Лучшие параметры XGBoost: {xgb_search.best_params_}")

        scalers[model_key] = scaler
        print("[Прогресс] Сохранение моделей, скейлера и энкодера...")
        for name, model in models.items():
            joblib.dump(model, f"models/{name}.joblib")
        joblib.dump(scaler, f"models/{model_key}_scaler.joblib")
        joblib.dump(encoder, f"models/{model_key}_encoder.joblib")
    else:
        print("[Прогресс] Загрузка существующих моделей...")
        models[f"{model_key}_rf"] = joblib.load(f"models/{model_key}_rf.joblib")
        models[f"{model_key}_xgb"] = joblib.load(f"models/{model_key}_xgb.joblib")
        scalers[model_key] = joblib.load(f"models/{model_key}_scaler.joblib")
        encoder = joblib.load(f"models/{model_key}_encoder.joblib")

    # Проверка точности и вероятностей на тестовых данных
    print("[Прогресс] Загрузка тестовых данных...")
    test_data = load_test_data("dataset_test.json")
    X_test_numeric, X_test_categorical_encoded, y_test, _, _ = prepare_features_and_target(test_data, mean_areas,
                                                                                           encoder=encoder,
                                                                                           fit_encoder=False)
    X_test_numeric_scaled = scalers[model_key].transform(X_test_numeric)
    X_test = hstack([X_test_categorical_encoded, csr_matrix(X_test_numeric_scaled)])

    metrics = {}
    for model_name in ['rf', 'xgb']:
        y_pred = models[f"{model_key}_{model_name}"].predict(X_test)
        y_proba = models[f"{model_key}_{model_name}"].predict_proba(X_test)[:, 1]
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        mean_proba = float(np.mean(y_proba))
        metrics[model_name] = {
            'Balanced Accuracy': balanced_acc,
            'Mean Probability (isCommercial=True)': mean_proba
        }
        print(f"[Результат] Balanced Accuracy {model_name.upper()}: {balanced_acc:.4f}")
        print(f"[Результат] Средняя вероятность (isCommercial=True) {model_name.upper()}: {mean_proba:.4f}")

    # Ансамбль
    rf_proba = models["general_rf"].predict_proba(X_test)[:, 1]
    xgb_proba = models["general_xgb"].predict_proba(X_test)[:, 1]
    ensemble_proba = (rf_proba + xgb_proba) / 2
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    ensemble_acc = balanced_accuracy_score(y_test, ensemble_pred)
    metrics['ensemble'] = {
        'Balanced Accuracy': ensemble_acc,
        'Mean Probability (isCommercial=True)': float(np.mean(ensemble_proba))
    }
    print(f"[Результат] Balanced Accuracy ENSEMBLE: {ensemble_acc:.4f}")
    print(f"[Результат] Средняя вероятность (isCommercial=True) ENSEMBLE: {ensemble_proba.mean():.4f}")

    data_store['X_test'] = X_test
    data_store['y_test'] = y_test
    print(f"[Прогресс] Итоговые модели: {list(models.keys())}")
    return True

load_or_train_models()

# Подготовка признаков для предсказания
def prepare_prediction_features(record, mean_areas, encoder):
    consumption_values = list(record.get("consumption", {}).values())
    if not consumption_values:
        print(f"[Предупреждение] Пустое потребление для записи {record.get('accountId', 'unknown')}, используется 0.")
        consumption_values = [0]

    consumption_values = cap_outliers(np.array(consumption_values))
    total_area = record.get("totalArea", mean_areas.get(record.get("buildingType", "Частный"), 100.0))
    total_area = max(total_area, 1.0)

    mean_consumption = np.mean(consumption_values) / total_area
    oct_to_apr = [v for k, v in record.get("consumption", {}).items() if int(k) in [1, 2, 3, 4, 10, 11, 12]]
    mean_oct_to_apr = np.mean(oct_to_apr) / total_area if oct_to_apr else mean_consumption
    median_consumption = np.median(consumption_values) / total_area
    max_consumption = np.max(consumption_values) / total_area
    min_consumption = np.min(consumption_values) / total_area

    consumption_variation = np.std(consumption_values) / (np.mean(consumption_values) + 1e-6)
    winter_consumption_ratio = np.mean(oct_to_apr) / (np.mean(consumption_values) + 1e-6) if oct_to_apr else 1.0
    residents_per_area = record.get("residentsCount", 1) / total_area
    high_consumption_flag = 1 if any(v > 3000 for v in oct_to_apr) else 0

    features_numeric = [
        mean_consumption,
        mean_oct_to_apr,
        median_consumption,
        max_consumption,
        min_consumption,
        consumption_variation,
        winter_consumption_ratio,
        residents_per_area,
        record.get("roomsCount", 1),
        record.get("residentsCount", 1),
        high_consumption_flag
    ]
    features_categorical = [[record.get("buildingType", "Частный")]]

    features_categorical_encoded = encoder.transform(features_categorical)
    features_numeric = np.array([features_numeric])
    features_numeric_scaled = scalers["general"].transform(features_numeric)
    features = hstack([features_categorical_encoded, csr_matrix(features_numeric_scaled)])

    print(f"[Прогресс] Подготовлены признаки для записи {record.get('accountId', 'unknown')}")
    return features

# Предсказание
@app.route('/predict', methods=['POST'])
def predict():
    print("[Прогресс] Получен запрос на /predict")
    try:
        data = request.get_json()
        if not data or "consumption" not in data:
            return jsonify({'error': 'Invalid input data'}), 400

        mean_areas = compute_mean_total_area(load_training_data("dataset_train.json"))
        features = prepare_prediction_features(data, mean_areas, encoder)

        # Предсказания от обеих моделей
        rf_proba = float(models["general_rf"].predict_proba(features)[0][1])
        xgb_proba = float(models["general_xgb"].predict_proba(features)[0][1])
        # Ансамбль
        ensemble_proba = (rf_proba + xgb_proba) / 2
        # Предсказания isCommercial
        is_commercial_rf = bool(models["general_rf"].predict(features)[0])
        is_commercial_xgb = bool(models["general_xgb"].predict(features)[0])
        is_commercial_ensemble = ensemble_proba > 0.5

        predictions = {
            'rf': rf_proba,
            'xgb': xgb_proba,
            'ensemble': ensemble_proba,
            'isCommercial_rf': is_commercial_rf,
            'isCommercial_xgb': is_commercial_xgb,
            'isCommercial': is_commercial_ensemble
        }

        feature_importance = {
            'rf': {name: float(imp) for name, imp in zip(feature_names, models["general_rf"].feature_importances_)},
            'xgb': {name: float(imp) for name, imp in zip(feature_names, models["general_xgb"].feature_importances_)}
        }
        predictions['feature_importance'] = feature_importance
        print(f"[Результат] Предсказание для записи: {predictions}")

        return jsonify(predictions)
    except Exception as e:
        print(f"[Ошибка] Ошибка предсказания: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

# Пакетное предсказание
@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    print("[Прогресс] Получен запрос на /batch-predict")
    try:
        if not request.is_json:
            print("[Ошибка] Тип содержимого запроса не JSON")
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        data = request.get_json()
        records = data.get('records', [])

        if not records:
            print("[Ошибка] Записи для пакетного предсказания не предоставлены")
            return jsonify({'error': 'No records provided'}), 400

        mean_areas = compute_mean_total_area(load_training_data("dataset_train.json"))
        results = []
        total_records = len(records)
        for i, record in enumerate(records):
            if i % 100 == 0:
                print(f"[Прогресс] Обработка записей: {i}/{total_records}")

            if not record or "consumption" not in record:
                print(f"[Предупреждение] Недействительная запись пропущена: {record}")
                results.append({
                    "accountId": record.get("accountId", 0),
                    "predictions": {
                        'rf': 0.0,
                        'xgb': 0.0,
                        'ensemble': 0.0,
                        'isCommercial_rf': False,
                        'isCommercial_xgb': False,
                        'isCommercial': False,
                        'feature_importance': {name: 0.0 for name in feature_names}
                    }
                })
                continue

            try:
                features = prepare_prediction_features(record, mean_areas, encoder)

                # Предсказания от обеих моделей
                rf_proba = float(models["general_rf"].predict_proba(features)[0][1])
                xgb_proba = float(models["general_xgb"].predict_proba(features)[0][1])
                # Ансамбль
                ensemble_proba = (rf_proba + xgb_proba) / 2
                # Предсказания isCommercial
                is_commercial_rf = bool(models["general_rf"].predict(features)[0])
                is_commercial_xgb = bool(models["general_xgb"].predict(features)[0])
                is_commercial_ensemble = ensemble_proba > 0.5

                predictions = {
                    'rf': rf_proba,
                    'xgb': xgb_proba,
                    'ensemble': ensemble_proba,
                    'isCommercial_rf': is_commercial_rf,
                    'isCommercial_xgb': is_commercial_xgb,
                    'isCommercial': is_commercial_ensemble
                }

                feature_importance = {
                    'rf': {name: float(imp) for name, imp in zip(feature_names, models["general_rf"].feature_importances_)},
                    'xgb': {name: float(imp) for name, imp in zip(feature_names, models["general_xgb"].feature_importances_)}
                }
                predictions['feature_importance'] = feature_importance

                results.append({
                    "accountId": record.get("accountId", 0),
                    "predictions": predictions
                })
            except Exception as e:
                print(f"[Ошибка] Ошибка обработки записи {record.get('accountId', 'unknown')}: {str(e)}")
                results.append({
                    "accountId": record.get("accountId", 0),
                    "predictions": {
                        'rf': 0.0,
                        'xgb': 0.0,
                        'ensemble': 0.0,
                        'isCommercial_rf': False,
                        'isCommercial_xgb': False,
                        'isCommercial': False,
                        'feature_importance': {name: 0.0 for name in feature_names}
                    }
                })

        print(f"[Прогресс] Пакетное предсказание завершено: обработано {len(results)} записей")
        return jsonify(results)
    except Exception as e:
        print(f"[Ошибка] Ошибка пакетного предсказания: {str(e)}")
        return jsonify({'error': f'Batch prediction error: {str(e)}'}), 500

# Метрики
@app.route('/metrics', methods=['GET'])
def metrics():
    print("[Прогресс] Получен запрос на /metrics")
    try:
        X_test = data_store['X_test']
        y_test = data_store['y_test']

        metrics = {}
        for model_name in ['rf', 'xgb']:
            y_pred = models[f"general_{model_name}"].predict(X_test)
            y_proba = models[f"general_{model_name}"].predict_proba(X_test)[:, 1]
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            mean_proba = float(np.mean(y_proba))
            metrics[model_name] = {
                'Balanced Accuracy': balanced_acc,
                'Mean Probability (isCommercial=True)': mean_proba
            }
            print(f"[Результат] Balanced Accuracy {model_name.upper()}: {balanced_acc:.4f}")
            print(f"[Результат] Средняя вероятность (isCommercial=True) {model_name.upper()}: {mean_proba:.4f}")

        # Ансамбль
        rf_proba = models["general_rf"].predict_proba(X_test)[:, 1]
        xgb_proba = models["general_xgb"].predict_proba(X_test)[:, 1]
        ensemble_proba = (rf_proba + xgb_proba) / 2
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        ensemble_acc = balanced_accuracy_score(y_test, ensemble_pred)
        metrics['ensemble'] = {
            'Balanced Accuracy': ensemble_acc,
            'Mean Probability (isCommercial=True)': float(np.mean(ensemble_proba))
        }
        print(f"[Результат] Balanced Accuracy ENSEMBLE: {ensemble_acc:.4f}")
        print(f"[Результат] Средняя вероятность (isCommercial=True) ENSEMBLE: {ensemble_proba.mean():.4f}")

        return jsonify(metrics)
    except Exception as e:
        print(f"[Ошибка] Ошибка вычисления метрик: {str(e)}")
        return jsonify({'error': f'Metrics error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)