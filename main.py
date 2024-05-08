import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from PIL import Image, ImageDraw, ImageOps

train = pd.read_csv("C:/Users/LENOVO/Computational Intelligence/Final/train.csv")
test = pd.read_csv("C:/Users/LENOVO/Computational Intelligence/Final/test.csv")
sample = pd.read_csv("C:/Users/LENOVO/Computational Intelligence/Final/sample_submission.csv")

y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) # Drop 'label' column
del train # free some space
y_train.value_counts()

model = RandomForestClassifier()

model.fit(X_train,y_train)

predict= model.predict(X_train)
predict

cm= metrics.confusion_matrix(y_train,model.predict(X_train))
cm

trainaccuracy= accuracy_score(y_train,model.predict(X_train))
trainaccuracy
print("Train Data Accuracy    :{} %".format(round((trainaccuracy*100),2)))

# Save the model to a file
joblib.dump(model, 'random_forest_model.pkl')

loaded_model = joblib.load('random_forest_model.pkl')

keras_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
])

keras_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

for i, layer in enumerate(keras_model.layers):
    if i == 0:
        layer.set_weights([loaded_model.coefs_[i], loaded_model.intercepts_[i]])
    elif i == 1:
        layer.set_weights([loaded_model.coefs_[i], loaded_model.intercepts_[i]])

model = tf.keras.models.load_model('random_forest_model.h5')
