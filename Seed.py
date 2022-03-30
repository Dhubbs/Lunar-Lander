from __future__ import division
import gym
import random

#This file is to be used to generate an initial seed for TrainNetwork.py
#This is an older file with sensitivity values found using a genetic algorithm



#base repersents enviroment variables
#Sensativity repersents a way to determine the importance of each enviroment variable
def react(base,Sensitivity):
	totals = []
	#builds a list with all readings about the enviroment multiplied by the passed in sensativity lelves
	for i in range(8):
		totals.extend([-base[i]*Sensitivity[i],base[i]*Sensitivity[i]])
	#finds the element that has the highest score meaning needs an action
	importantIndex = totals.index(max(totals))
	#lookup table describing which rocket to fire based on most important attribute
	#This has been manually figured out
	lookup = [3,1,2,0,3,1,2,0,3,1,1,3,3,1]
	return lookup[importantIndex]

env = gym.make('LunarLander-v2')

f = open("out.txt","a")
#run 200 simulations
for i in range(200): 
	done = False
	s = env.reset()
	tot = 0
	#run a single simulation
	while(not done):
		#record enviroment before
		before = s
		reaction = react(s,[-1.4820802315616515, 1.1638440825458152, 4.573096457594461, 4.223255280588565, -2.7458770404281547, 7.847426286227093, -0.2219472169364971, 4.0073635318388945])
		s,score,done,a = env.step(reaction)
		tot += score
		#if we have a positive score, write it
		if score>0:
			f.write(",".join(before)+","+str(reaction)+"\n")
f.close()
