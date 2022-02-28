# Lunar Lander
In this project, I chose to use a neural network in an unconventional method to solve the lunar lander challenge.

Lunar Lander is a task in which you are given an environment and data regarding things such as the locations and velocity of a rocket as it approaches the ground, and you must develop a system to manage the up, left, and right motors in order to safely land the rocket.

I've previously done this task using a genetic algorithm and sensitivity levels to determine when motors should be fired, but this time I've chosen to utilise a neural network. The reason I think it's unusual is that I've utilised it in a way that requires the network to be constantly trained, tested, and retrained in order to improve its performance. The way it works is that there must first be some seed data representing the simulation conditions and desired reaction, after which the network is trained on this data and begins simulating the outcomes. The simulation conditions + the reaction are saved if the decision results in a better "score" as determined by the simulation. The simulation is run 50 times, and the network is retrained on this new data, continuing the cycle.


This is a computationally intensive method of solving this problem, with good results not appearing until >1,000,000 records of positive activities have been recorded. I used a virtual machine on GCP for training and was able to attain a consistent passing score after around 5 hours of training.

[The Program in Action](https://www.youtube.com/watch?v=2kQWBPc8SOU)
