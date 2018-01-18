import gym
import time
import os
#Dylan Hubble
#Used with OpenAi gym Lunar lander problem



#function used to determine if firing the left or right engine is needed
#depending on the current state of the the ship
def problem(temp,sensitivity):
	amount = 0
	if abs(temp[0])>abs(temp[4]):
		print "distance"
		amount = temp[0]
	else:
		print "angle"
		amount = -temp[4]
	if amount>0:
		return 1
	else:
		return 3



env = gym.make('LunarLander-v2')

base = []
done = False
total = 0

#runs the simulation 5 times
for j in range(5):
	#sets up tracking for scoring of how well each simulation goes
	total += 100
	base.append(total)

	#resets the enviroment for the next run
	simulation = env.reset()

	#sets up a counter variable
	i = 0

	#resets the score tracker
	total = 0


	done = False

	while(not done):
		env.render()
		i+=1


		#tuning variable used to determine how sensitive to be
		# with tilt,vertical speed and drift
		xpositionTune = 110
		angleTune = 100
		VspeedTune = 540
		sensitivityTune = 120


		simulation[0] = simulation[0]*xpositionTune 
		simulation[4] = simulation[4]*angleTune  
		simulation[3] = -simulation[3]*VspeedTune


		# move possibilitys  1 right thruster, 2 upwards,3 left thrusters ,0 nothing
		move = 0

		#every second step through checks if the tilt needs to be changed
		if i%2:
			move =  problem(simulation,sensitivityTune)
			
		#determines if the upwards thrust needs to be applied above a certain value
		elif simulation[3]>120:
			move = 2


		#clears the screen
		os.system("clear")

		#displays information to see how the ship is doing
		print ""
		print "distance:" +str(simulation[1])
		print "angle:" + str(-simulation[4])
		print ""

		#updates the enviroment with the action taken
		temp = env.step(move)   

		#sets simulation to be the state of the ship
		simulation = temp[0]

		#collects the score
		total += temp[1]	

		#determines if the simulation is complete	
		done = temp[2]

		#prints the total score so far
		print total


#clears the screen and displays the total cummulative score
os.system("clear")
print "final score:" + str(sum(base))