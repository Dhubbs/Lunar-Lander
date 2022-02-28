from __future__ import division
import gym
from mllib import *

#depends on a "seed" of data
#continuously retrains a neural net and uses it to generate more data
#uses a neural net to build a greedy approach to the lunar lander problem and optimizes every step for max score
#will need >1,000,000 data points to start seeing progress, and considerable compute
#Chose to not render as I used a GCP vm that took ~5 hours for reasonable results

#loads the enviroment
env = gym.make('LunarLander-v2')
#repeats the train test simulation 10 times, this value can be increased to allow for longer execution and more training
for j in range(10):
    #retrains the model with the data stored in the out.txt file
	#fir is a function from mllib.py
	accuracy,model = fit((100,50,10),"out.txt")
	#accuracy of the newly trained model
	print("Accuracy:"+ str(accuracy))
	#stores the currently trained model so that the program can be run continuosly and exited at any time with the most recent model saved
	save(model,"first.sav")
	#open our out.txt base data file to append more data
	f = open("out.txt",'a')
	avg = 0
	#runs 50 simulated interations
	for i in range(50):
		#set up to run an individual simulation
		done = False
		s = env.reset()
		tot = 0
		#start the simulation
		while(not done):
			#saves the current state of the simulation so that it can be saved if a model
			before = s
			#predicts which rocket to fire based on the current state of the simulation
			reaction = predict(model,s)
			#applys the prediction and steps the simulation forward
			s,score,done,a = env.step(reaction)
			#keeps a rolling total of the current score for the simulation
			tot += score
			#if the action we took had a positive impact on our total score we want to record the enviroment + the action taken
			if score>0.3:
				#writes enviroment + reaction
				for j in before:
					f.write(str(j)+",")
				f.write(str(reaction) + "\n")
			#if we have completed simulation add total score to 50 simulation sum
			if done == True:
				avg += tot
	#prints our avg score across the 50 simulaions
	print("AVG:" + str(avg/50.0))
	#closes out.txt so that it can be read from for retraining
	f.close()
