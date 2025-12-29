import os
import joblib
import numpy as np
import warnings
from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask_cors import CORS

# =========================
# 0) 忽略警告（可选）
# =========================
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# 1) 路径配置（以当前 app.py 所在目录为根目录）
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model.updated.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.updated.joblib")

# =========================
# 2) 加载模型与预处理器（启动时加载一次）
# =========================
model = None
scaler = None

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Model loading failed: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded successfully.")
except Exception as e:
    print(f"❌ Scaler loading failed: {e}")

# =========================
# 3) Flask 初始化
# =========================
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# 只给 /api/* 开 CORS（更安全）
CORS(app, resources={r"/api/*": {"origins": "*"}})

# =========================
# 4) 表单取值：转 float（空值/非法 -> default）
# =========================
def get_feature_value(field_name: str, default_value: float = 0.0) -> float:
    value = request.form.get(field_name)
    if value is None:
        return default_value

    value = str(value).strip()
    if value == "":
        return default_value

    # 兼容 Yes/No
    if value in ("Yes", "1", "yes", "Y", "y", "True", "true"):
        return 1.0
    if value in ("No", "0", "no", "N", "n", "False", "false"):
        return 0.0

    # 兼容小数逗号
    value = value.replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return default_value

# =========================
# 5) 模型输入特征
#    逻辑：前 9 个标准化，Baseline_VAS 不标准化（最后一列原样拼回去）
# =========================
NUM_FEATURES = ["Age", "PCS", "MCS", "PSQI", "LY#", "MO#", "ALB", "Glu", "A/G"]
CAT_FEATURE = "Baseline_VAS"  # 不标准化
MODEL_FEATURES = NUM_FEATURES + [CAT_FEATURE]  # 总共 10 个

# 表单字段名 -> 模型特征名（确保与你 GitHub Pages 的 index.html input name 一致）
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
    "baseline_vas": "Baseline_VAS",
}

def build_feature_vector() -> np.ndarray:
    """
    构建 (1,10) 特征向量，顺序严格按 MODEL_FEATURES
    """
    feat_map = {f: 0.0 for f in MODEL_FEATURES}
    for form_key, feat_name in FORM_TO_FEATURE.items():
        feat_map[feat_name] = get_feature_value(form_key, 0.0)

    x = np.array([feat_map[f] for f in MODEL_FEATURES], dtype=float).reshape(1, -1)
    return x

def scale_features_baseline_unscaled(x_10: np.ndarray) -> np.ndarray:
    """
    - NUM_FEATURES：按 scaler 的 mean/scale 标准化（按列名对齐）
    - Baseline_VAS：不标准化，原样拼回去
    """
    if scaler is None:
        raise RuntimeError("Scaler is not loaded.")

    cols = list(getattr(scaler, "feature_names_in_", []))
    if not cols:
        raise RuntimeError("Scaler does not have feature_names_in_.")

    # 检查 NUM_FEATURES 都在 scaler 里
    missing_num = [c for c in NUM_FEATURES if c not in cols]
    if missing_num:
        raise ValueError(f"Numeric features missing in scaler columns: {missing_num}")

    # ✅ 不再依赖 x_10 的“位置切片”，改成按 MODEL_FEATURES 显式取列
    feat_index = {name: i for i, name in enumerate(MODEL_FEATURES)}

    x_num = np.array([x_10[:, feat_index[f]] for f in NUM_FEATURES], dtype=float).T  # (1,9)
    x_cat = x_10[:, [feat_index[CAT_FEATURE]]]  # (1,1)

    idx_num = [cols.index(c) for c in NUM_FEATURES]
    means = scaler.mean_[idx_num]
    scales = scaler.scale_[idx_num]

    x_num_scaled = (x_num - means) / scales
    x_final = np.concatenate([x_num_scaled, x_cat], axis=1)  # (1,10)
    return x_final

def predict_from_request():
    """
    统一预测逻辑：返回 pred(int), proba(float), x_final
    """
    if model is None:
        raise RuntimeError("Model not loaded.")
    if scaler is None:
        raise RuntimeError("Scaler not loaded.")

    x_10 = build_feature_vector()
    x_final = scale_features_baseline_unscaled(x_10)

    proba = float(model.predict_proba(x_final)[:, 1][0])
    pred = int(model.predict(x_final)[0])
    return pred, proba, x_final

# =========================
# 6) 页面路由（Render 打开也能用）
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    传统页面跳转（用于你 Render 直接打开网页提交表单）
    """
    try:
        pred, proba, _ = predict_from_request()
        return redirect(url_for(
            "result",
            prediction_class=str(pred),
            prediction_proba=f"{proba:.4f}"
        ))
    except Exception as e:
        print(f"Prediction logic failed: {e}")
        return f"Prediction failed: {e}", 400

@app.route("/result")
def result():
    prediction_class = request.args.get("prediction_class", "N/A")
    prediction_proba = request.args.get("prediction_proba", "N/A")

    if prediction_class == "1":
        result_text = "High Risk"
    elif prediction_class == "0":
        result_text = "Low Risk"
    else:
        result_text = "Abnormal prediction results"

    return render_template(
        "results.html",
        prediction_class_text=result_text,
        prediction_proba=prediction_proba
    )

# =========================
# 7) API 路由（给 GitHub Pages 调用）
# =========================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    GitHub Pages 前端 fetch 调用这个接口，返回 JSON
    """
    try:
        pred, proba, _ = predict_from_request()
        return jsonify({
            "prediction_class": pred,
            "prediction_proba": round(proba, 4),
            "risk_text": "High Risk" if pred == 1 else "Low Risk"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 可选：健康检查（Render 监控/你自己测试）
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })

# =========================
# 8) 启动（本地用；Render 用 gunicorn 启动，不走这里）
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
