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


def predict_from_request():
    """
    统一预测逻辑：返回 pred(int), proba(float), x_final
    """
    if model is None:
        raise RuntimeError("Model not loaded.")
    if scaler is None:
        raise RuntimeError("Scaler not loaded.")

    input_features = [
        get_feature_value('age'),
        get_feature_value('pcs_score'),
        get_feature_value('mcs_ics_score'),
        get_feature_value('psqi_score'),
        get_feature_value('ly_count'),
        get_feature_value('mono_count'),
        get_feature_value('alb'),
        get_feature_value('glu'),
        get_feature_value('a_g'),
        get_feature_value('baseline_vas_score')
    ]

    final_features = np.array(input_features).reshape(1, -1)

    means = scaler.mean_[[0, 4, 5, 6, 8, 9, 39, 42, 50]]
    scales = scaler.scale_[[0, 4, 5, 6, 8, 9, 39, 42, 50]]

    x_num = final_features[:, :9]
    x_cat = final_features[:, [-1]]

    x_num_scaled = (x_num - means) / scales
    x_final = np.concatenate([x_num_scaled, x_cat], axis=1)

    # 3. 进行预测
    prediction_proba = float(model.predict_proba(x_final)[:, 1])  # 取 PHN 阳性的概率 (第二列)
    prediction_class = int(model.predict(x_final)[0])  # 取预测的类别 (0 或 1)

    return prediction_class, prediction_proba, x_final

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
