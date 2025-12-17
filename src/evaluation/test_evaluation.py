from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def evaluate_test(model, X_test_vec, y_test, model_name="Model"):
    """
    Evaluate model performance on TEST data
    """

    y_test_pred = model.predict(X_test_vec)

    print(f"\n📊 {model_name} — TEST Evaluation")
    print("-" * 50)

    acc = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {acc:.4f}\n")

    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred))

    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
