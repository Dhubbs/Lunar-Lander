import random
import math
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import csv
import pickle

clf = 0

def fit(x,y,z,filename):
    #change to x,y,z
    clf = MLPClassifier(solver = 'lbfgs' , alpha = 1, hidden_layer_sizes = (x,y,z))
    
    print "Reading files..."
    with open(filename, 'rb') as f:
        reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
        vals = list(reader)
    
    
    print "Cleaning Data..."
    y = [0 for i in range(len(vals))]
    
    for i in range(len(vals)):
        y[i] = vals[i][len(vals[i])-1]
        del vals[i][len(vals[i])-1]

    lb = LabelEncoder()
    y = lb.fit_transform(y)
    print "Splitting Data..."
    vals = np.asarray(vals, dtype=np.float32)

    
    X_train, X_test, y_train, y_test = train_test_split(vals,y,test_size=0.10)
    print "Training..."
    clf.fit(X_train,y_train)

    total = len(X_test)
    right = 0
    rightconfi = 0

    corr60 = 0
    time60 = 1
    corr75 = 0
    time75 = 1
    corr90 = 0
    time90 = 1

    print "Evaluating..."
    for i in range(len(X_test)):
        #print clf.predict_proba([X_test[i]])
        #print clf.predict([X_test[i]])
        if(max(clf.predict_proba([X_test[i]])[0])>0.6):
            time60 += 1
            if(clf.predict([X_test[i]]) == y_test[i]):
                corr60 += 1
        if(max(clf.predict_proba([X_test[i]])[0])>0.75):
            time75 += 1
            if(clf.predict([X_test[i]]) == y_test[i]):
                corr75 += 1

        if(max(clf.predict_proba([X_test[i]])[0])>0.8):
            time90 += 1
            if(clf.predict([X_test[i]]) == y_test[i]):
                corr90 += 1

        if clf.predict([X_test[i]]) == y_test[i]:
            right += 1


    #print str(right/float(total)) + " out of " + str(float(total)) + " normal"
    #print str(corr60/float(time60)) + " out of " + str(float(time60)) + " 60%"
    #print str(corr75/float(time75)) + " out of " + str(float(time75)) + " 75%"
    #print str(corr90/float(time90)) + " out of " + str(float(time90)) + " 80%"

    return right/float(total),clf

def save(model,filename):
    pickle.dump(model,open(filename, 'wb'))
def load(filename):
    return pickle.load(open(filename, 'rb'))
def predict(model,vals):
    #print "Splitting Data..."
    #vals = np.asarray(vals, dtype=np.float32)
    #return model.predict(vals.reshape(1,-1))
    #for i in vals:
    #print str(i) + " , " + str(model.predict([i]))
    #print "Splitting Data..."
    vals = [vals]
    vals = np.asarray(vals, dtype=np.float32)
    #vals = scale(vals,axis = 0,with_mean=False,with_std=True,copy=True)
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



#accuracy,model = fit(100,50,100,"out.txt")
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

