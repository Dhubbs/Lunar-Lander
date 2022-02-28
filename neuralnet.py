from __future__ import division
import gym
import random
import random
import math
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import csv
import pickle
from mllib import *


#depends on a "seed" of data
#continuously retrains a neural net and uses it to generate more data
#


env = gym.make('LunarLander-v2')

for j in range(10):
   
    #builds a new 
    accuracy,model = fit(10,10,4,"out.txt")
    print("Accuracy:"+ str(accuracy))
    
    save(model,"first.sav")

    f = open("out.txt",'a')
    accuracy = 0
    avg = 0
    for i in range(50): 	
    	    done = False
	    s = env.reset()
	    tot = 0
	    while(not done):
		#env.render()
		before = s
		#print(list(s))
                #print(s)
		reaction = predict(model,s)
		s,score,done,a = env.step(reaction)
		tot += score
		if 1==1:
			for j in before:
				f.write(str(j)+",")
			f.write(str(reaction) + "\n")
		if done == True:
			print i
			print tot
			avg += tot
    print("AVG:" + str(avg/50.0))
    f.close()
