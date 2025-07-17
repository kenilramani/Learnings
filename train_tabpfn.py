import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
try:
    from tabpfn import TabPFNClassifier
except ImportError:  # pragma: no cover - optional dependency
    TabPFNClassifier = None

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric mean absolute percentage error."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_true - y_pred) / denom) * 100

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["amount"] = df["amount"].round().astype("Int64")
    return df

def prepare_dataset(df: pd.DataFrame, product_id: int):
    df_filtered = df[df["productId"] == product_id].copy()

    # bin continuous target for classification
    y_cont = df_filtered["amount"]
    y_class, bins = pd.qcut(y_cont, q=10, labels=False, duplicates="drop", retbins=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    features = [
        "taxableAmount",
        "sgstAmount",
        "cgstAmount",
        "rate",
        "pts",
        "qty",
        "billAmount",
    ]
    X = df_filtered[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train_class, y_test_class, y_train_cont, y_test_cont = train_test_split(
        X_scaled, y_class, y_cont, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train_class, y_test_cont, bin_centers

def train_model(X_train, y_train, device: str = "cpu"):
    if TabPFNClassifier is None:
        raise ImportError("tabpfn is required for training")
    model = TabPFNClassifier(device=device)
    model.fit(X_train, y_train)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train TabPFN model to predict amount")
    parser.add_argument("--data", required=True, help="Path to the CSV dataset")
    parser.add_argument("--product_id", type=int, required=True, help="Product ID to filter on")
    parser.add_argument("--device", default="cpu", help="Device for TabPFN (cpu or cuda)")
    args = parser.parse_args()

    df = load_data(args.data)
    X_train, X_test, y_train_class, y_test_cont, bin_centers = prepare_dataset(df, args.product_id)
    model = train_model(X_train, y_train_class, device=args.device)
    pred_class = model.predict(X_test)
    y_pred = bin_centers[pred_class]

    print(f"MAPE: {mape(y_test_cont.to_numpy(), y_pred):.2f}%")
    print(f"SMAPE: {smape(y_test_cont.to_numpy(), y_pred):.2f}%")


if __name__ == "__main__":
    main()
