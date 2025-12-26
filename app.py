import os
import joblib
import numpy as np
from flask import Flask, request, render_template, redirect, url_for

# -------------------------
# 1) 路径配置（以当前 app.py 所在目录为根目录）
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'final_model.updated.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.updated.joblib')

# -------------------------
# 2) 加载模型与预处理器
# -------------------------
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Scaler loading failed: {e}")
    scaler = None

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))


# -------------------------
# 3) 表单取值：转 float（空值/非法 -> default）
# -------------------------
def get_feature_value(field_name, default_value=0.0):
    value = request.form.get(field_name)
    if value is None:
        return default_value
    value = str(value).strip()
    if value == '':
        return default_value

    # 你有 Yes/No 或 gender 这种情况可保留（目前本模型输入都是数值）
    if value in ('Yes', '1'):
        return 1.0
    if value in ('No', '2'):
        return 0.0

    try:
        return float(value)
    except ValueError:
        return default_value


# -------------------------
# 4) 模型输入特征顺序（必须固定）
# -------------------------
MODEL_FEATURES = ["Age", "PCS", "MCS", "PSQI", "LY#", "MO#", "ALB", "Glu", "A/G", "CO2"]

# 表单字段名 -> 模型特征名
FORM_TO_FEATURE = {
    "age": "Age",
    "pcs_score": "PCS",
    "mcs_ics_score": "MCS",
    "psqi_score": "PSQI",
    "ly_count": "LY#",
    "mono_count": "MO#",
    "alb": "ALB",
    "glu": "Glu",
    "a_g": "A/G",
    "co2": "CO2",
}


def build_feature_vector():
    # 先收集成 dict，确保不缺字段
    feat_map = {f: 0.0 for f in MODEL_FEATURES}
    for form_key, feat_name in FORM_TO_FEATURE.items():
        feat_map[feat_name] = get_feature_value(form_key, 0.0)

    # 按固定顺序拼成 (1,10)
    x = np.array([feat_map[f] for f in MODEL_FEATURES], dtype=float).reshape(1, -1)
    return x


def scale_features(x_10):
    """
    使用 scaler.updated.joblib 的 mean_/scale_，按列名对齐标准化 10 个特征。
    scaler 是在 53 列上 fit 的，所以必须从 scaler 里取这 10 列对应的参数。
    """
    if scaler is None:
        raise RuntimeError("Scaler is not loaded.")

    cols = list(getattr(scaler, "feature_names_in_", []))
    if not cols:
        raise RuntimeError("Scaler does not have feature_names_in_.")

    missing = [c for c in MODEL_FEATURES if c not in cols]
    if missing:
        raise ValueError(f"These features are not in scaler training columns: {missing}")

    idx = [cols.index(c) for c in MODEL_FEATURES]
    means = scaler.mean_[idx]
    scales = scaler.scale_[idx]

    x_scaled = (x_10 - means) / scales
    return x_scaled


# -------------------------
# 5) 路由
# -------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "模型未加载，无法预测。", 500
    if scaler is None:
        return "预处理器未加载，无法预测。", 500

    try:
        x_10 = build_feature_vector()      # (1,10)
        x_final = scale_features(x_10)     # (1,10)

        prediction_proba = model.predict_proba(x_final)[:, 1]
        prediction_class = model.predict(x_final)[0]

        return redirect(url_for(
            'result',
            prediction_class=str(prediction_class),
            prediction_proba=f"{prediction_proba[0]:.4f}"
        ))

    except Exception as e:
        print(f"Prediction logic failed: {e}")
        return f"Prediction failed: {e}", 400


@app.route('/result')
def result():
    prediction_class = request.args.get('prediction_class', 'N/A')
    prediction_proba = request.args.get('prediction_proba', 'N/A')

    if prediction_class == '1':
        result_text = "High Risk"
    elif prediction_class == '0':
        result_text = "Low Risk"
    else:
        result_text = "Abnormal prediction results"

    return render_template(
        'results.html',
        prediction_class_text=result_text,
        prediction_proba=prediction_proba
    )


if __name__ == '__main__':
    app.run(debug=True)
