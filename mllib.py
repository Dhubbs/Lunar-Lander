
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import pickle

#standard library I have developed to allow for ease of use of sklearn classifier
#built on reading csv files with last column as the Target and all other columns as the Predictors
#saving and loading is done through pickling the model to allow for use elsewhere

clf = 0
#x = tuple describing size of hidden layers
#filename = csv data containing training data
def fit(x,filename):
    clf = MLPClassifier(solver = 'lbfgs' , alpha = 1, hidden_layer_sizes = x)
    print("Reading files...")

    with open(filename, 'rb') as f:
        vals = list(csv.reader(f,quoting=csv.QUOTE_NONNUMERIC))
    print("Cleaning Data...")
    y = [0] * len(vals)
    #pop last column
    #creates a list y containing the Target + removes Target from rest of data
    for i in range(len(vals)):
        y[i] = vals[i][len(vals[i])-1]
        del vals[i][len(vals[i])-1]

    #allows Target to be non numeric
    lb = LabelEncoder()
    y = lb.fit_transform(y)

    print("Splitting Data...")

    #to numpy array
    vals = np.asarray(vals, dtype=np.float32)
    #90 10 train test split
    X_train, X_test, y_train, y_test = train_test_split(vals,y,test_size=0.10)
    print("Training...")
    clf.fit(X_train,y_train)
    total = len(X_test)
    
    right = 0
    print("Evaluating...")
    for i in range(len(X_test)):
        if clf.predict([X_test[i]]) == y_test[i]:
            right += 1
    #returns accuracy,model object
    return right/float(total),clf

def save(model,filename):
    pickle.dump(model,open(filename, 'wb'))
def load(filename):
    return pickle.load(open(filename, 'rb'))
#takes in model,vals to predict on
def predict(model,vals):
    vals = np.asarray([vals], dtype=np.float32)
    for i in vals:
        return model.predict([i])[0]
