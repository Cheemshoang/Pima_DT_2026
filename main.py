import numpy as np
import argparse
from sklearn.metrics import accuracy_score
from src.dt import DecisionTree
from src.xgb import XGBoostTree, XGBoostModel
from src.rf import RandomForest
from utils.loader import get_data


def parse_args():
    parser = argparse.ArgumentParser(description="Run model")
    subparsers = parser.add_subparsers(dest="model", required=True)
    
    dt_parser = subparsers.add_parser('dt')
    dt_parser.add_argument("--costfunc", "-i", type=str, choices=["gini", "entropy"], default="gini", help="gini or entropy")
    dt_parser.add_argument("--max_depth", "-m", type=int, default=5)
    dt_parser.add_argument("--min_split", "-n", type=int, default=3)
    
    rf_parser = subparsers.add_parser('rf')
    rf_parser.add_argument("--n_trees", type=int, default=10)
    rf_parser.add_argument("--max_depth", "-m", type=int, default=5)
    rf_parser.add_argument("--min_split", "-n", type=int, default=3)
    rf_parser.add_argument("--n_features", type=int, default=10)
    
    xgb_parser = subparsers.add_parser('xgb')
    xgb_parser.add_argument("--n_trees", type=int, default=50)
    xgb_parser.add_argument("--learning_rate", type=float, default=0.1)
    xgb_parser.add_argument("--max_depth", "-m", type=int, default=3)
    xgb_parser.add_argument("--lambda_reg", type=float, default=1.0)

    return parser.parse_args()
def main():
    args = parse_args()
    kwargs = vars(args).copy()
    model_type = kwargs.pop('model')
    X_train, y_train, X_val, X_test, y_test = get_data()
    
    if args.model == "dt":
        cart_model = DecisionTree(**kwargs)
        cart_model.fit(X_train, y_train)

        id3_model = DecisionTree(**kwargs)
        id3_model.fit(X_train, y_train)

        cart_preds = cart_model.predict(X_test.to_numpy())
        id3_preds = id3_model.predict(X_test.to_numpy())

        cart_acc = np.sum(cart_preds == y_test.to_numpy().flatten()) / len(y_test)
        id3_acc = np.sum(id3_preds == y_test.to_numpy().flatten()) / len(y_test)

        print(f"CART Accuracy:   {cart_acc * 100 :.2f}%")
        print(f"ID3 Accuracy:    {id3_acc * 100:.2f}%")
    
    if args.model == 'xgb':
        xgb_model = XGBoostModel(**kwargs)
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)

        xgb_acc = np.sum(xgb_preds == y_test.to_numpy().flatten()) / len(y_test)
        print(f"XGBoost Accuracy: {xgb_acc *100:.2f}%")
    
    if args.model == "rf":
        rf_model = RandomForest(**kwargs)   
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        rf_acc = np.sum(rf_preds == y_test.to_numpy().flatten()) / len(y_test)
        print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")


if __name__ == "__main__":
    main()