import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks

df = pd.read_csv("weather_data.csv")

df["target"] = (df["tmax_next_day"] > df["tmax_next_day"].median()).astype(int)

X = df.drop(["target", "date"], axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
base_acc = accuracy_score(y_test, DecisionTreeClassifier().fit(X_train, y_train).predict(X_test))
print("1) Базовая accuracy:", base_acc)

counts = Counter(y)
majority = max(counts, key=counts.get)
minority = min(counts, key=counts.get)
keep_n = int(counts[majority] * 0.1)

df_major = df[df["target"] == majority].sample(keep_n, random_state=42)
df_minor = df[df["target"] == minority]
df_imbal = pd.concat([df_major, df_minor], ignore_index=True)

X2 = df_imbal.drop(["target", "date"], axis=1)
y2 = df_imbal["target"]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, shuffle=False)
imbal_acc = accuracy_score(y2_test, DecisionTreeClassifier().fit(X2_train, y2_train).predict(X2_test))
print("2) Accuracy после создания дисбаланса:", imbal_acc)

methods = {
    "SMOTE": SMOTE(),
    "ADASYN": ADASYN(),
    "TomekLinks": TomekLinks()
}

for name, method in methods.items():
    X_res, y_res = method.fit_resample(X_train, y_train)
    acc = accuracy_score(y_test, DecisionTreeClassifier().fit(X_res, y_res).predict(X_test))
    print(f"3) {name}: accuracy = {acc}")
