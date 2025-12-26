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

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# --- 1. 配置 ---
DATA_FILE = '外部数据集11.2.xlsx'
MODEL_FILE = 'final_model.updated.joblib'
PREPROCESS_FILE = 'scaler.updated.joblib'
TARGET_COLUMN = 'PHN'  # 目标列名

# 10 个最终输入到模型的特征（顺序必须与训练最终喂给模型一致）
KEY_FEATURES = ["Age", "PCS", "MCS", "PSQI", "LY#", "MO#", "ALB", "Glu", "A/G","CO2"]


# =========================================================================
# === 修正函数：解决 joblib 文件的编码问题 ===
# =========================================================================
def safe_joblib_load(file_path: str):
    try:
        return joblib.load(file_path)
    except UnicodeDecodeError as e:
        if 'utf-8' in str(e) and 'invalid start byte' in str(e):
            print(f"⚠️ 警告：加载 {file_path} 时检测到 UTF-8 编码错误，尝试使用 'latin-1' 编码。")
            return joblib.load(file_path, mmap_mode=None, encoding='latin1')
        raise
    except Exception as e:
        if "No such file or directory" in str(e):
            raise FileNotFoundError(f"文件未找到: {file_path}")
        raise


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """清理列名：转 str、去首尾空格、把多个空格压成单个空格（可选）"""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


print("--- 外部验证：模型准确率测试开始 ---")

# --- 2. 加载模型、预处理器、数据 ---
try:
    final_model = safe_joblib_load(MODEL_FILE)
    preprocess = safe_joblib_load(PREPROCESS_FILE)
    data = pd.read_excel(DATA_FILE)
    data = clean_columns(data)

    print("模型和预处理器加载成功。")
    print(f"数据集加载成功，共 {len(data)} 条记录。")

except Exception as e:
    print(f"错误：加载文件时发生异常: {e}")
    raise SystemExit(1)
# --- DEBUG: 打印 preprocess 训练时使用的特征列名 ---
print("当前 preprocess 文件：", PREPROCESS_FILE)
print("\n--- preprocess 训练时的特征列名 ---")
if hasattr(preprocess, "feature_names_in_"):
    print(list(preprocess.feature_names_in_))
    print("Baseline_VAS in scaler.feature_names_in_ ?",
          "Baseline_VAS" in preprocess.feature_names_in_)
    print("特征数量:", len(preprocess.feature_names_in_))
else:
    print("preprocess 没有 feature_names_in_ 属性（训练时可能用的是 numpy）")

# --- 3. 强制执行预处理 + 预测（关键：先53维预处理，再取10维给模型） ---
try:
    # 3.1 标签
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"数据集中找不到目标列 {TARGET_COLUMN}，现有列：{list(data.columns)[:30]} ...")
    y_true = data[TARGET_COLUMN].values

    # 3.2 拿到 preprocess fit 时使用的 53 列列名（最稳）
    if hasattr(preprocess, "feature_names_in_"):
        ALL_FEATURES = list(preprocess.feature_names_in_)
    else:
        # 如果你的 preprocess 是纯 numpy fit 的，就没有 feature_names_in_
        # 那你必须在这里手动提供训练时的 53 个列名列表（顺序必须一致）
        raise ValueError(
            "preprocess 没有 feature_names_in_。\n"
            "说明你 fit preprocess 时可能传的是 numpy（没有列名），或版本/对象不带该属性。\n"
            "请手动提供训练时 scaler.fit 使用的 53 个特征列名列表 ALL_FEATURES（顺序一致）。"
        )

    # 3.3 确保外部数据集含有 ALL_FEATURES，不存在的列补 NaN
    for col in ALL_FEATURES:
        if col not in data.columns:
            data[col] = np.nan

    # 3.4 取出 53 列（严格按 ALL_FEATURES 顺序）
    X_53_df = data[ALL_FEATURES].copy()

    # 3.5 数值化 + 缺失处理（要尽量与你训练 preprocess 时一致）
    #     注意：如果你的训练时用了更复杂的预处理（比如 OneHot、分组填充等），
    #     这里应该复刻同样策略。你目前保存的 preprocess 是 scaler，所以用数值化+中位数填充。
    for c in ALL_FEATURES:
        X_53_df[c] = pd.to_numeric(X_53_df[c], errors="coerce")

    X_53_df = X_53_df.fillna(X_53_df.median(numeric_only=True))

    # 3.6 先对 53 维做 transform（传 DataFrame，避免“无列名”warning）
    X_53_processed = preprocess.transform(X_53_df)  # (N, 53)

    # 3.7 从处理后的 53 维中抽取模型需要的 10 维（按名字找索引，避免顺序错）
    missing_key = [f for f in KEY_FEATURES if f not in ALL_FEATURES]
    if missing_key:
        raise ValueError(f"KEY_FEATURES 有列不在 preprocess 的训练列中：{missing_key}")

    key_idx = [ALL_FEATURES.index(f) for f in KEY_FEATURES]
    X_10_for_model = X_53_processed[:, key_idx]  # (N, 10)

    # --- 调试信息（强烈建议保留到你跑通为止） ---
    print("\n--- 维度/对象检查 ---")
    print("外部数据总列数:", data.shape[1])
    print("ALL_FEATURES(预处理期望)数量:", len(ALL_FEATURES))
    print("KEY_FEATURES(模型输入)数量:", len(KEY_FEATURES))
    print("X_53_df shape:", X_53_df.shape)
    print("X_53_processed shape:", X_53_processed.shape)
    print("X_10_for_model shape:", X_10_for_model.shape)
    print("preprocess type:", type(preprocess))
    print("preprocess expects n_features_in_:", getattr(preprocess, "n_features_in_", None))
    print("model type:", type(final_model))
    print("model expects n_features_in_:", getattr(final_model, "n_features_in_", None))

    # 3.8 模型预测（模型吃10维）
    y_prob = final_model.predict_proba(X_10_for_model)[:, 1]
    y_pred = final_model.predict(X_10_for_model)

except Exception as e:
    print(f"\n 预测或预处理失败。")
    print(f"原始错误: {e}")
    raise SystemExit(1)

# --- 4. 结果计算和输出 ---
print("\n--- 结果计算 ---")

cm = confusion_matrix(y_true, y_pred)
if cm.shape != (2, 2):
    print("❌ 混淆矩阵不是二分类 2x2：", cm.shape)
    print(cm)
    raise SystemExit(1)

TN, FP, FN, TP = cm.ravel()

accuracy = accuracy_score(y_true, y_pred)
sensitivity = recall_score(y_true, y_pred)  # 召回率
specificity = TN / (TN + FP) if (TN + FP) else 0
ppv = precision_score(y_true, y_pred)
npv = TN / (TN + FN) if (TN + FN) else 0
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
