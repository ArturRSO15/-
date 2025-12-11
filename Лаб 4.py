import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from collections import Counter


df = pd.read_csv("weather_data.csv")  

df["target"] = (df["tmax_next_day"] > df["tmax_next_day"].median()).astype(int)
X = df.drop(["target", "date"], axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = DecisionTreeClassifier().fit(X_train, y_train)
base_acc = accuracy_score(y_test, model.predict(X_test))
print("1) Базовая accuracy:", base_acc)

counts = Counter(y)
minority = min(counts, key=counts.get)
remove_n = int(counts[minority] * 0.1)

df_imbal = df.copy()
df_imbal = df_imbal[df_imbal["target"] != minority].append(
    df[df["target"] == minority].sample(remove_n)
)

X2 = df_imbal.drop(["target", "date"], axis=1)
y2 = df_imbal["target"]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, shuffle=False)
acc_imbal = accuracy_score(y2_test, DecisionTreeClassifier().fit(X2_train, y2_train).predict(X2_test))
print("2) Accuracy после искусственного дисбаланса:", acc_imbal)

methods = {
    "SMOTE": SMOTE(),
    "ADASYN": ADASYN(),
    "TomekLinks": TomekLinks()
}

for name, method in methods.items():
    X_res, y_res = method.fit_resample(X_train, y_train)
    acc = accuracy_score(y_test, DecisionTreeClassifier().fit(X_res, y_res).predict(X_test))
    print(f"3) {name}: accuracy = {acc}")
