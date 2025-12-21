import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(csv_path, test_size=0.3, random_state=42):
    df = pd.read_csv(csv_path)

    X = df["User Query"]
    y_intent = df["User Intent"]
    y_team = df["Assigned Team"]

    X_train, X_test, y_intent_train, y_intent_test = train_test_split(
        X, y_intent, test_size=test_size, random_state=random_state
    )

    # Align team labels with same split
    y_team_train = y_team.loc[y_intent_train.index]
    y_team_test = y_team.loc[y_intent_test.index]

    return X_train, X_test, y_intent_train, y_intent_test, y_team_train, y_team_test
