import pandas as pd
import numpy as np

data = pd.read_csv("Spam Data.csv")

from sklearn.model_selection import StratifiedShuffleSplit
x = data.iloc[:, :-1]
y = data["Spam"]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_data, test_data in split.split(x, y, data["word_freq_business:"]):
    strat_train_set = data.loc[train_data]
    strat_test_set = data.loc[test_data]

data = strat_train_set.copy()
corr_matrix = data.corr()
corr_matrix["Spam"].sort_values(ascending=False)


from sklearn.impute import SimpleImputer
data = strat_train_set.drop("Spam", axis=1)
data_labels = strat_train_set["Spam"].copy()
imputer = SimpleImputer(strategy="median")
imputer.fit(data)
X = imputer.transform(data)
data_tr= pd.DataFrame(X, columns=data.columns)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

data_num_tr = my_pipeline.fit_transform(data)
"""
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(data_num_tr, data_labels)
some_data = data.iloc[:5]
some_labels = data_labels[:5]
prepared_data = my_pipeline.transform(some_data)
"""
"""
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(data_num_tr, data_labels)
some_data = data.iloc[:5]
some_labels = data_labels[:5]
prepared_data = my_pipeline.transform(some_data)"""


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(data_num_tr, data_labels)
some_data = data.iloc[:5]
some_labels = data_labels[:5]
prepared_data = my_pipeline.transform(some_data)

"""
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(data_num_tr, data_labels)
some_data = data.iloc[:5]
some_labels = data_labels[:5]
prepared_data = my_pipeline.transform(some_data)"""

from sklearn.metrics import mean_squared_error
data_predection = model.predict(data_num_tr)
lin_mse = mean_squared_error(data_labels, data_predection)
lin_rmse = np.sqrt(lin_mse)

#Cross Validation
from sklearn.model_selection import cross_val_score

score = cross_val_score(model, data_num_tr, data_labels, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-score)


def print_Score(scorese):
    print("Score: ", scorese)
    print("mean: ", scorese.mean())
    print("std: ", scorese.std())

print(print_Score(rmse_scores))

X_test = strat_test_set.drop("Spam", axis = 1)
Y_test = strat_test_set["Spam"].copy()

X_test_prepared = my_pipeline.transform(X_test)
final_Prepared = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_Prepared)
final_rmse = np.sqrt(final_mse)

print(final_rmse)
from joblib import dump, load

dump(model, 'Spam.joblib')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(Y_test, final_Prepared)
precision = precision_score(Y_test, final_Prepared)
recall = recall_score(Y_test, final_Prepared)
f1 = f1_score(Y_test, final_Prepared)

# Print the evaluation metrics
def metrix_iq():
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")

print(metrix_iq())
