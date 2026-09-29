"""
Зээлийн оноо (300–850) таамаглах загвар сургах скрипт.

Ажиллуулах:  python train.py
Үр дүн:
    model.joblib   – сургасан загвар + текст→тоо хөрвүүлэлт + баганын дараалал
    encoders.json  – текст→тоо хөрвүүлэлтийн хүснэгт (хүн уншихад зориулсан хуулбар)
    metrics.json   – MAE, R2 болон цэвэрлэгээний тайлан

app.py нь энэ файлаас clean_data, encode_features, decide функцуудыг импортолж
ашигладаг тул сургалт болон таамаглал ЯГ ИЖИЛ дүрмээр явагдана.
"""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = "credit_data - credit_data.csv"
MODEL_PATH = "model.joblib"
ENCODERS_PATH = "encoders.json"
METRICS_PATH = "metrics.json"

TARGET = "Зээлийн оноо"
SCORE_MIN, SCORE_MAX = 300, 850

# Хувийн мэдээлэл (PII) – загварт ХЭРЭГЛЭХГҮЙ (өгөгдлийг хамгийн бага хэрэглэх зарчим)
PII_COLUMNS = ["Хэрэглэгчийн дугаар", "Нэр", "Имэйл"]

NUMERIC_COLUMNS = [
    "Нас",
    "Ажилласан жил",
    "Сарын орлого",
    "Зээлийн үлдэгдэл",
    "Өр орлогын харьцаа",
    "Хугацаа хэтрүүлсэн тоо",
    "Идэвхтэй зээлийн тоо",
    "Зээлийн түүх сараар",
]

# ---------------------------------------------------------------------------
# Текст → тоо хөрвүүлэлт. Энэ хүснэгтийг model.joblib болон encoders.json-д
# хадгалдаг тул app.py шинэ өгөгдлийг сургалттай яг ижил кодоор хөрвүүлнэ.
# ---------------------------------------------------------------------------
# Боловсрол нь дараалалтай (ordinal) тул 0,1,2,3 гэж шатлалаар кодлоно
EDUCATION_MAP = {"Бүрэн дунд": 0, "Бакалавр": 1, "Магистр": 2, "Доктор": 3}
# Барьцаа хөрөнгө нь тийм/үгүй (binary)
COLLATERAL_MAP = {"Байхгүй": 0, "Байгаа": 1}
# Ажлын төрөл нь дараалалгүй тул one-hot (тус бүрдээ 0/1 багана) болгоно
JOB_TYPES = ["Хувийн хэвшил", "Төрийн албан хаагч", "Хувиараа", "Ажилгүй", "Оюутан"]

ENCODERS = {
    "Боловсрол": EDUCATION_MAP,
    "Барьцаа хөрөнгө": COLLATERAL_MAP,
    "Ажлын төрөл": JOB_TYPES,
}

# Шийдвэрийн босго
APPROVE_MIN = 670   # 670+      → Зөвшөөрөх
REVIEW_MIN = 580    # 580–669   → Нягтлах, 580-аас доош → Татгалзах


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Алдаатай мөрийг хасаж, юу хасагдсаныг тайлан болгон буцаана."""
    report = {"эхний_мөр": len(df)}

    # 1) Сарын орлого хоосон эсвэл 0 бол хасна
    income = pd.to_numeric(df["Сарын орлого"], errors="coerce")
    bad_income = income.isna() | (income <= 0)
    report["орлого_хоосон_эсвэл_0"] = int(bad_income.sum())

    # 2) Нас 150 ба түүнээс дээш бол боломжгүй утга гэж үзэж хасна
    age = pd.to_numeric(df["Нас"], errors="coerce")
    bad_age = age.isna() | (age >= 150)
    report["нас_150_дээш"] = int((bad_age & ~bad_income).sum())

    # 3) Өр орлогын харьцаа сөрөг бол хасна
    ratio = pd.to_numeric(df["Өр орлогын харьцаа"], errors="coerce")
    bad_ratio = ratio.isna() | (ratio < 0)
    report["сөрөг_харьцаа"] = int((bad_ratio & ~bad_income & ~bad_age).sum())

    clean = df[~(bad_income | bad_age | bad_ratio)].copy()
    report["үлдсэн_мөр"] = len(clean)
    report["нийт_хасагдсан"] = report["эхний_мөр"] - report["үлдсэн_мөр"]
    return clean, report


def encode_features(df: pd.DataFrame, encoders: dict = ENCODERS) -> pd.DataFrame:
    """Текст багануудыг хадгалсан хөрвүүлэлтээр тоо болгож, загварын оролт бэлдэнэ."""
    X = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce").copy()

    X["Боловсрол"] = df["Боловсрол"].map(encoders["Боловсрол"])
    X["Барьцаа хөрөнгө"] = df["Барьцаа хөрөнгө"].map(encoders["Барьцаа хөрөнгө"])
    for job in encoders["Ажлын төрөл"]:
        X[f"Ажил_{job}"] = (df["Ажлын төрөл"] == job).astype(int)

    # Мэдэгдэхгүй ангилал ирвэл чимээгүй таамаглахгүй – алдаа өгнө
    unknown = X["Боловсрол"].isna() | X["Барьцаа хөрөнгө"].isna() | ~df["Ажлын төрөл"].isin(encoders["Ажлын төрөл"])
    if unknown.any():
        raise ValueError(f"Мэдэгдэхгүй текст утгатай {int(unknown.sum())} мөр байна (Боловсрол / Ажлын төрөл / Барьцаа хөрөнгө).")
    return X


def decide(score: float) -> str:
    """Онооноос шийдвэр гаргана."""
    if score >= APPROVE_MIN:
        return "Зөвшөөрөх"
    if score >= REVIEW_MIN:
        return "Нягтлах"
    return "Татгалзах"


def main():
    raw = pd.read_csv(DATA_PATH, encoding="utf-8")
    df, report = clean_data(raw)
    print("Цэвэрлэгээ:", report)

    X = encode_features(df)
    y = df[TARGET]

    # 80% сургалт, 20% шалгалт. random_state тогтмол → давтан сургахад ижил үр дүн
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Таамаглалыг 300–850 хооронд хязгаарлаж хэмжинэ
    pred = model.predict(X_test).clip(SCORE_MIN, SCORE_MAX)
    metrics = {
        "MAE": round(float(mean_absolute_error(y_test, pred)), 2),
        "R2": round(float(r2_score(y_test, pred)), 4),
        "сургалтын_мөр": len(X_train),
        "шалгалтын_мөр": len(X_test),
        "цэвэрлэгээ": report,
        "сургасан_огноо": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"MAE = {metrics['MAE']}   R2 = {metrics['R2']}")

    # Загвар, хөрвүүлэлт, баганын дарааллыг НЭГ файлд хадгална
    joblib.dump({"model": model, "encoders": ENCODERS, "feature_columns": list(X.columns)}, MODEL_PATH)
    with open(ENCODERS_PATH, "w", encoding="utf-8") as f:
        json.dump(ENCODERS, f, ensure_ascii=False, indent=2)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Хадгалсан: {MODEL_PATH}, {ENCODERS_PATH}, {METRICS_PATH}")


if __name__ == "__main__":
    main()
