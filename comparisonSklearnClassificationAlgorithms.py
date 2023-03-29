import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

plt.style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')

wine_dataset = datasets.load_wine()

print("Keys of wine_dataset:\n", wine_dataset.keys())

print("Feature names:\n", wine_dataset['feature_names'])

print("Target names:", wine_dataset['target_names'])

print("Shape of data:", wine_dataset['data'].shape)

print("Shape of target:", wine_dataset['target'].shape)
print("Target:\n", wine_dataset['target'])

X = wine_dataset.data
y = wine_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#shape of the X_train and y_train
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

#shape of the X_test and y_test
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))

results = [] #list for append cross_val_scores
names = [] 
scoring = 'accuracy'

kfold = model_selection.KFold(n_splits=10, shuffle=True)

for name, model in models:
  cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f" % (name, cv_results.mean())
  print(msg)

fig = plt.figure(figsize=(10,10))
fig.suptitle('Comparison of sklearn classification algorithms')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()