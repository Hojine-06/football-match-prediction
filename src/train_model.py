import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, ROOT
from evaluate_model import evaluate

TRAIN = ["1516", "1617", "1718", "1819", "1920", "2021", "2122"]
VAL = ["2223", "2324"]
TEST = ["2425"]

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    df["season"] = df["season"].astype(str).str.zfill(4)

    features = [c for c in df.columns if c.startswith(("home_", "away_", "diff_"))
                and c not in ["home_team", "away_team", "home_goals", "away_goals"]
                and not any(k in c for k in ["shots", "corners", "fouls"])]
    tr, va, te = (df[df.season.isin(s)] for s in (TRAIN, VAL, TEST))
    Xtr, ytr = tr[features], tr["result"]
    Xva, yva = va[features], va["result"]
    Xte, yte = te[features], te["result"]

    models = {
        "baseline": DummyClassifier(strategy="prior"),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                n_jobs=-1, random_state=42),
        "gradient_boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                            max_iter=200, random_state=42),
    }

    val_results = []
    for name, m in models.items():
        m.fit(Xtr, ytr)
        val_results.append(evaluate(m, Xva, yva, name))
    val_df = pd.DataFrame(val_results).sort_values("log_loss")
    print("\nVALIDATION\n", val_df.round(4))

    best = val_df.iloc[0]["model"]
    print(f"\nMeilleur modèle (log-loss validation) : {best}")

    # Test final : une seule fois, sur la saison 2024-2025
    test_results = []
    for name in ["baseline", best]:
        m = models[name]
        test_results.append(evaluate(m, Xte, yte, name, save_cm=True, verbose=True))
    test_df = pd.DataFrame(test_results)
    print("\nTEST\n", test_df.round(4))

    (ROOT / "reports").mkdir(exist_ok=True)
    val_df.to_csv(ROOT / "reports" / "validation_results.csv", index=False)
    test_df.to_csv(ROOT / "reports" / "test_results.csv", index=False)
    joblib.dump(models[best], MODELS_DIR / "best_model.joblib")

if __name__ == "__main__":
    main()