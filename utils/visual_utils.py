import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from utils.custom_exceptions import CustomException

def plot_feature_importance(model, feature_names=None, top_n=20, save_path=None, title="Feature Importance"):
    try:
        importance = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importance))]
        
        sorted_idx = np.argsort(importance)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_idx)), importance[sorted_idx], align='center')
        plt.xticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx], rotation=45, ha='right')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
    except Exception as e:
        raise CustomException(e)


def plot_predicted_vs_actual(y_true, y_pred, save_path=None):
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs Actual")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Predicted vs Actual plot saved to {save_path}")
        else:
            plt.show()
    except Exception as e:
        raise CustomException(e)


def plot_residuals(y_true, y_pred, save_path=None):
    try:
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Residual plot saved to {save_path}")
        else:
            plt.show()
    except Exception as e:
        raise CustomException(e)


def plot_error_distribution(y_true, y_pred, save_path=None):
    try:
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True, bins=30)
        plt.xlabel("Residuals")
        plt.title("Residuals Distribution")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Error distribution plot saved to {save_path}")
        else:
            plt.show()
    except Exception as e:
        raise CustomException(e)

