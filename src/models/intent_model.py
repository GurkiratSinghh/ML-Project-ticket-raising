from sklearn.linear_model import LogisticRegression

def train_intent_model(X_train_vec, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    return model
