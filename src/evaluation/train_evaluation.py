from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def evaluate_train(model, X_train_vec, y_train, model_name="Model"):
    """
    Evaluate model performance on TRAIN data
    """

    y_train_pred = model.predict(X_train_vec)

    print(f"\n📊 {model_name} — TRAIN Evaluation")
    print("-" * 50)

    acc = accuracy_score(y_train, y_train_pred)
    print(f"Train Accuracy: {acc:.4f}\n")

    print("Train Classification Report:")
    print(classification_report(y_train, y_train_pred))

    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))
