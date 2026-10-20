import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             log_loss, classification_report)
from config import FIGURES_DIR

LABELS = ["H", "D", "A"]


def evaluate(model, X, y, name, save_cm=False, verbose=False):
    pred = model.predict(X)
    proba = model.predict_proba(X)

    if verbose:
        print(f"\n=== {name} ===")
        print(classification_report(y, pred, labels=LABELS, zero_division=0))

    if save_cm:
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y, pred, labels=LABELS, ax=ax, colorbar=False)
        ax.set_title(f"Matrice de confusion : {name}")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"confusion_{name}.png", dpi=150)
        plt.close(fig)

    return {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "log_loss": log_loss(y, proba, labels=list(model.classes_)),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_draw": f1_score(y, pred, labels=["D"], average=None, zero_division=0)[0],
    }