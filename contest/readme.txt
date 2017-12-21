Team 4: Sean Fox, Amal Morais, Kushal Tiwari

	The goal of the capture the flag Pacman game, based on a UC Berkeley assignment, was to design agents that could intelligently both Pacman and ghosts and apply
	coordinated team-based strategies in order to consume more food than the opponent's team while maintaining one’s distance from the enemy and defending your own
	side. 
	
	The offensive agent uses approximated Q values that has features which determine several pieces of key information: The distance to the closest enemy, closest 
	food pellet, the closest power-up pellet, if they are entering into a tunnel, and what direction the agent is moving in. When in danger, it has the foresight 
	to determine if it is a better option to return to the user’s origin point and start searching for food from there instead of continuing on towards the next 
	food source.

	For the defensive agent, a particle filtering algorithm is implemented that would approximate an invader’s current location with respect to them. After 
	computing this, it is careful to never let the distance to the enemy get too close; even stopping and reversing directions when they are too near one 
	another. 

	Errors that came up through the course of the assignment mainly dealt with depth and getting stuck in a particular section of the grid, causing Pacman to loop
	around until he is caught. We had some success in reducing the randomness that was our results, however there is still a large amount of room for chance to 
	occur in or code that can skew the results. On average, we tend to result in 8-10 points after successfully running the code, however there are some outliers 
	that do occur.