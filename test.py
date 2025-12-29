import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    auc as skl_auc,
    precision_recall_curve
)
from sklearn.exceptions import InconsistentVersionWarning

# =========================
# 0) 忽略警告（可选）
# =========================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# =========================
# 1) 配置
# =========================
DATA_FILE = '外部数据集11.2.xlsx'
MODEL_FILE = 'final_model.updated.joblib'
SCALER_FILE = 'scaler.updated.joblib'
TARGET_COLUMN = 'PHN'  # 目标列名（二分类：0/1）

# 模型最终输入的 10 个特征
KEY_FEATURES = ["Age", "PCS", "MCS", "PSQI", "LY#", "MO#", "ALB", "Glu", "A/G", "CO2"]

# =========================
# 2) 工具函数
# =========================
def safe_joblib_load(file_path: str):
    """加载 joblib"""
    try:
        return joblib.load(file_path)
    except UnicodeDecodeError as e:
        if 'utf-8' in str(e) and 'invalid start byte' in str(e):
            print(f"⚠️ 警告：加载 {file_path} UTF-8 编码异常，尝试 latin-1")
            return joblib.load(file_path, mmap_mode=None, encoding='latin1')
        raise
    except Exception as e:
        if "No such file or directory" in str(e):
            raise FileNotFoundError(f"文件未找到: {file_path}")
        raise

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """清理列名：转 str、去首尾空格、把多个空格压成单个空格"""
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return out

def ensure_binary_labels(y: np.ndarray) -> np.ndarray:
    """保证 y 为 0/1（若你的标签是 'Yes/No' 等，请在此扩展映射）"""
    y = pd.Series(y).copy()
    # 常见字符串映射（按需扩展）
    y = y.replace({
        "0": 0, "1": 1,
        "No": 0, "Yes": 1,
        "N": 0, "Y": 1,
        "False": 0, "True": 1
    })
    y = pd.to_numeric(y, errors="coerce")
    if y.isna().any():
        bad_n = int(y.isna().sum())
        raise ValueError(f"目标列 {TARGET_COLUMN} 有 {bad_n} 条无法转成数值(0/1) 的标签。请先清洗标签。")
    y = y.astype(int).values
    uniq = set(np.unique(y))
    if not uniq.issubset({0, 1}):
        raise ValueError(f"目标列 {TARGET_COLUMN} 不是二分类0/1，实际取值: {sorted(list(uniq))}")
    return y

print("--- External Validation: Full-Scaling (Same style as Code A) ---")

# 3.1 加载模型、scaler、数据
try:
    final_model = safe_joblib_load(MODEL_FILE)
    scaler = safe_joblib_load(SCALER_FILE)
    data = pd.read_excel(DATA_FILE)
    data = clean_columns(data)

    print("模型与 scaler 加载成功。")
    print(f"数据集加载成功，共 {len(data)} 条记录。")


except Exception as e:
    print(f"错误：加载文件时发生异常: {e}")
    raise SystemExit(1)

# 3.2 读取标签
if TARGET_COLUMN not in data.columns:
    print(f"数据集中找不到目标列 {TARGET_COLUMN}，现有列示例：{list(data.columns)[:30]} ...")
    raise SystemExit(1)

y_true = ensure_binary_labels(data[TARGET_COLUMN].values)

# 3.3 获取 scaler 训练时使用的所有特征列（例如 53 列）
if not hasattr(scaler, "feature_names_in_"):
    raise SystemExit(
        "scaler 没有 feature_names_in_。\n"
        "说明你 fit scaler 时可能传的是 numpy（没有列名）或对象不带该属性。\n"
        "解决：训练时用 DataFrame.fit(...) 保存 scaler；或手动提供 ALL_FEATURES 列名列表。"
    )

ALL_FEATURES = list(scaler.feature_names_in_)
print("\n--- scaler 训练时的特征列名（前20个预览）---")
print(ALL_FEATURES[:20])
print("特征数量:", len(ALL_FEATURES))

# 3.4 确保外部数据包含 ALL_FEATURES，不存在的补 NaN
for col in ALL_FEATURES:
    if col not in data.columns:
        data[col] = np.nan

# 3.5 取出训练期望的全部特征列（严格按 ALL_FEATURES 顺序）
X_53_df = data[ALL_FEATURES].copy()

# 3.6 数值化（scaler 只能吃数值）
for c in ALL_FEATURES:
    X_53_df[c] = pd.to_numeric(X_53_df[c], errors="coerce")

# 3.7 缺失填充：用中位数
X_53_df = X_53_df.fillna(X_53_df.median(numeric_only=True))
# 3.8 统一执行 scaler.transform
X_53_processed = scaler.transform(X_53_df)  # shape: (N, len(ALL_FEATURES))
# 3.9 从处理后的全部特征中抽取模型需要的 10 个特征
missing_key = [f for f in KEY_FEATURES if f not in ALL_FEATURES]
if missing_key:
    raise SystemExit(f"KEY_FEATURES 有列不在 scaler 的训练列中：{missing_key}")

key_idx = [ALL_FEATURES.index(f) for f in KEY_FEATURES]
X_10_for_model = X_53_processed[:, key_idx]  # shape: (N, 10)

# 3.10 维度检查
print("\n--- 维度检查 ---")
print("X_53_df shape:", X_53_df.shape)
print("X_53_processed shape:", X_53_processed.shape)
print("X_10_for_model shape:", X_10_for_model.shape)
print("scaler type:", type(scaler))
print("scaler expects n_features_in_:", getattr(scaler, "n_features_in_", None))
print("model type:", type(final_model))
print("model expects n_features_in_:", getattr(final_model, "n_features_in_", None))

# 3.11 模型预测
try:
    y_prob = final_model.predict_proba(X_10_for_model)[:, 1]
    y_pred = final_model.predict(X_10_for_model)
except Exception as e:
    print("预测失败：", e)
    raise SystemExit(1)


print("\n--- 结果计算 ---")

cm = confusion_matrix(y_true, y_pred)
if cm.shape != (2, 2):
    print("混淆矩阵不是二分类 2x2：", cm.shape)
    print(cm)
    raise SystemExit(1)

TN, FP, FN, TP = cm.ravel()

accuracy = accuracy_score(y_true, y_pred)
sensitivity = recall_score(y_true, y_pred)  # TPR
specificity = TN / (TN + FP) if (TN + FP) else 0.0
ppv = precision_score(y_true, y_pred)       # Precision
npv = TN / (TN + FN) if (TN + FN) else 0.0
f1 = f1_score(y_true, y_pred)

auc_val = roc_auc_score(y_true, y_prob)
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
pr_auc = skl_auc(recall_vals, precision_vals)

print("\n[External Validation - Final Model Only]")
print("Model:       ExtraTrees")
print(f"Accuracy:    {accuracy:.3f}")
print(f"AUC:         {auc_val:.3f}")
print(f"PR-AUC:      {pr_auc:.3f}")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"PPV:         {ppv:.3f}")
print(f"NPV:         {npv:.3f}")
print(f"F1:          {f1:.3f}")

print("\nConfusion Matrix:")
print(f"[[{TN:^3}  {FP:^3}]")
print(f" [ {FN:^3}  {TP:^3}]]")

print("\n--- 测试完成 ---")
