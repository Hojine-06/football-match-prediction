import json
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, ROOT
from evaluate_model import evaluate

TRAIN = ["1516", "1617", "1718", "1819", "1920", "2021", "2122"]
VAL = ["2223", "2324"]
TEST = ["2425"]
SEED = 42


def make_models():
    return {
        "baseline": DummyClassifier(strategy="prior"),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                n_jobs=-1, random_state=SEED),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
            early_stopping=False, random_state=SEED),
    }


def main():
    for d in (MODELS_DIR, FIGURES_DIR, ROOT / "reports"):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"],
                     dtype={"season": str}, index_col="match_id")
    features = json.loads((PROCESSED_DIR / "feature_columns.json").read_text())

    tr, va, te = (df[df.season.isin(s)] for s in (TRAIN, VAL, TEST))
    trva = df[df.season.isin(TRAIN + VAL)]
    print(f"train={len(tr)} | validation={len(va)} | test={len(te)} | train+val={len(trva)}")

    # 1) Comparaison sur la validation (entraînement sur train uniquement)
    val_rows = []
    for name, model in make_models().items():
        model.fit(tr[features], tr["result"])
        val_rows.append(evaluate(model, va[features], va["result"], name))
    val_df = pd.DataFrame(val_rows).sort_values("log_loss").reset_index(drop=True)
    print("\nVALIDATION (2022-2024)\n", val_df.round(4).to_string(index=False))

    best = val_df[val_df.model != "baseline"].iloc[0]["model"]
    print(f"\nModèle retenu (log-loss validation la plus basse) : {best}")

    # 2) Ré-entraînement sur train+val, test final unique sur 2024-2025
    test_rows, fitted = [], {}
    for name, model in make_models().items():
        model.fit(trva[features], trva["result"])
        fitted[name] = model
        test_rows.append(evaluate(model, te[features], te["result"], name,
                                  save_cm=name in ("baseline", best),
                                  verbose=name in ("baseline", best)))
    test_df = pd.DataFrame(test_rows)
    print("\nTEST (2024-2025)\n", test_df.round(4).to_string(index=False))

    # 3) Sauvegardes
    reports = ROOT / "reports"
    val_df.to_csv(reports / "validation_results.csv", index=False)
    test_df.to_csv(reports / "test_results.csv", index=False)

    best_model = fitted[best]
    proba = pd.DataFrame(best_model.predict_proba(te[features]),
                         columns=[f"p_{c}" for c in best_model.classes_], index=te.index)
    out = te[["date", "league", "season", "home_team", "away_team", "result"]].join(proba)
    out["pred"] = best_model.predict(te[features])
    out.to_csv(reports / "test_predictions.csv")

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    (MODELS_DIR / "best_model_name.txt").write_text(best)


if __name__ == "__main__":
    main()