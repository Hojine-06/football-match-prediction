import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, log_loss,
                             classification_report, ConfusionMatrixDisplay)
from config import FIGURES_DIR

def evaluate(model, X, y, name, save_cm=False, verbose=False):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "log_loss": log_loss(y, proba, labels=model.classes_),
    }
    if verbose:
        print(f"\n=== {name} ===")
        print(classification_report(y, pred, zero_division=0))
    if save_cm:
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax, colorbar=False)
        ax.set_title(f"Matrice de confusion : {name}")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"confusion_{name}.png", dpi=150)
        plt.close(fig)
    return metrics