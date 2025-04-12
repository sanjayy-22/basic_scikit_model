from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

#dataset
iris = load_iris()
X, y = iris.data, iris.target

#train
model = RandomForestClassifier()
model.fit(X, y)

#save
joblib.dump(model, 'model.pkl')

#joblib is used to pickle or group the project like a pipeline
