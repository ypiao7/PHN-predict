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
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded successfully.")
except Exception as e:
    print(f"❌ Scaler loading failed: {e}")
    scaler = None

# =========================
# 3) Flask 初始化
# =========================
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
CORS(app)  # ✅ 允许 GitHub Pages 等跨域访问 /api/predict

# =========================
# 4) 表单取值：转 float（空值/非法 -> default）
# =========================
def get_feature_value(field_name, default_value=0.0):
    """
    从表单中读取字段值：
    - None/空字符串 -> default
    - Yes/No -> 1/0
    - 其余尝试 float
    """
    value = request.form.get(field_name)
    if value is None:
        return default_value

    value = str(value).strip()
    if value == "":
        return default_value

    if value in ("Yes", "1"):
        return 1.0
    if value in ("No", "2"):
        return 0.0

    # 兼容用户输入 "3,5" 这种逗号小数
    value = value.replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return default_value

# =========================
# 5) 模型输入特征顺序（必须固定）
#    逻辑：前 9 个标准化，Baseline_VAS 不标准化（最后一列原样拼回去）
# =========================
NUM_FEATURES = ["Age", "PCS", "MCS", "PSQI", "LY#", "MO#", "ALB", "Glu", "A/G"]
CAT_FEATURE = "Baseline_VAS"  # 不标准化
MODEL_FEATURES = NUM_FEATURES + [CAT_FEATURE]  # 总共10个

# 表单字段名 -> 模型特征名
# ⚠️ index.html 的 input name 必须与左侧一致
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
    "baseline_vas": "Baseline_VAS",  # ✅ 你的表单字段建议用 baseline_vas
}

def build_feature_vector():
    """
    构建 (1,10) 特征向量，顺序严格按 MODEL_FEATURES
    """
    feat_map = {f: 0.0 for f in MODEL_FEATURES}
    for form_key, feat_name in FORM_TO_FEATURE.items():
        feat_map[feat_name] = get_feature_value(form_key, 0.0)

    x = np.array([feat_map[f] for f in MODEL_FEATURES], dtype=float).reshape(1, -1)
    return x

def scale_features_like_code2(x_10: np.ndarray) -> np.ndarray:
    """
    前 9 个数值特征：按 scaler 的 mean/scale 标准化（按列名对齐）
    Baseline_VAS：不做标准化，原样拼回去
    """
    if scaler is None:
        raise RuntimeError("Scaler is not loaded.")

    cols = list(getattr(scaler, "feature_names_in_", []))
    if not cols:
        raise RuntimeError("Scaler does not have feature_names_in_.")

    # 只要求 NUM_FEATURES 在 scaler 列名中存在
    missing_num = [c for c in NUM_FEATURES if c not in cols]
    if missing_num:
        raise ValueError(f"Numeric features missing in scaler training columns: {missing_num}")

    idx_num = [cols.index(c) for c in NUM_FEATURES]
    means = scaler.mean_[idx_num]
    scales = scaler.scale_[idx_num]

    x_num = x_10[:, :len(NUM_FEATURES)]
    x_cat = x_10[:, [len(NUM_FEATURES)]]

    x_num_scaled = (x_num - means) / scales
    x_final = np.concatenate([x_num_scaled, x_cat], axis=1)
    return x_final

# =========================
# 6) 页面路由（本地可用）
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    保留你原来页面跳转式预测：用于本地/传统方式
    """
    if model is None:
        return "模型未加载，无法预测。", 500
    if scaler is None:
        return "预处理器未加载，无法预测。", 500

    try:
        x_10 = build_feature_vector()
        x_final = scale_features_like_code2(x_10)

        prediction_proba = model.predict_proba(x_final)[:, 1]
        prediction_class = model.predict(x_final)[0]

        return redirect(url_for(
            "result",
            prediction_class=str(int(prediction_class)),
            prediction_proba=f"{float(prediction_proba[0]):.4f}"
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
    GitHub Pages / 前端 fetch 调用这个接口，返回 JSON
    """
    if model is None:
        return jsonify({"error": "model not loaded"}), 500
    if scaler is None:
        return jsonify({"error": "scaler not loaded"}), 500

    try:
        x_10 = build_feature_vector()
        x_final = scale_features_like_code2(x_10)

        proba = float(model.predict_proba(x_final)[:, 1][0])
        pred = int(model.predict(x_final)[0])

        return jsonify({
            "prediction_class": pred,
            "prediction_proba": round(proba, 4),
            "risk_text": "High Risk" if pred == 1 else "Low Risk"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# =========================
# 8) 启动
# =========================
if __name__ == "__main__":
    # Render 部署时不会走这里；本地运行才用
    app.run(host="0.0.0.0", port=5000, debug=True)
