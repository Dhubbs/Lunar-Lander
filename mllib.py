
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import csv
import pickle

#standard library I have developed to allow for ease of use of sklearn classifier
#built on reading csv files with last column as the Target and all other columns as the predictors
#saving and loading is done through pickling the model to allow for use else where
#add scaling

clf = 0
def fit(x,filename):
    clf = MLPClassifier(solver = 'lbfgs' , alpha = 1, hidden_layer_sizes = x)
    print("Reading files...")
    with open(filename, 'rb') as f:
        reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
        vals = list(reader)
    print("Cleaning Data...")
    y = [0 for i in range(len(vals))]
    #pop last column
    for i in range(len(vals)):
        y[i] = vals[i][len(vals[i])-1]
        del vals[i][len(vals[i])-1]

    #allows Target to be non numeric
    lb = LabelEncoder()
    y = lb.fit_transform(y)

    print("Splitting Data...")
    vals = np.asarray(vals, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(vals,y,test_size=0.10)
    print("Training...")
    clf.fit(X_train,y_train)
    total = len(X_test)
    right = 0
    print("Evaluating...")
    for i in range(len(X_test)):
        if clf.predict([X_test[i]]) == y_test[i]:
            right += 1
    #returns accuralcy + model
    return right/float(total),clf

def save(model,filename):
    pickle.dump(model,open(filename, 'wb'))
def load(filename):
    return pickle.load(open(filename, 'rb'))
def predict(model,vals):
    vals = [vals]
    vals = np.asarray(vals, dtype=np.float32)
    for i in vals:
        return model.predict([i])[0]


#builds your model
#in1 = first layer size
#in2 = second layer size
#in3 = third layer size
#in4 = training data file
#training data must have what you are predicting too as last column
#out1 = accuracy of model
#out2 = model



#accuracy,model = fit((100,50,100),"out.txt")
#print(accuracy)





#predictis on Data
#in1 = model object
#in2 = file of data to predict on
#predict data must be one row with (number of columns of training data - 1) columns
#prints the data as well as the prediction
#can be modified to return it
#predict(model,"pre.csv")

#saves a model for later use
#in1 = the neural net model
#in2 = save file name, can be any extension
#save(model,"first.sav")



#loads a model from file
#in1 = location of a previously built model
#out1 = returns a neural net object
#load("first.sav")


#pip install -U scikit-learn
#pip install numpy

