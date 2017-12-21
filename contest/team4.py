# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
 
 
from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Agent
import game
from util import nearestPoint
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions
import keyboardAgents
from util import Counter
from distanceCalculator import manhattanDistance


 
 #################
 # Team creation #
 #################
 
def createTeam(firstIndex, secondIndex, isRed, first = 'OffensiveAgent', second = 'DefensiveAgent'):
     """
     This function should return a list of two agents that will form the
     team, initialized using firstIndex and secondIndex as their agent
     index numbers.  isRed is True if the red team is being created, and
     will be False if the blue team is being created.
     
     As a potentially helpful development aid, this function can take
     additional string-valued keyword arguments ("first" and "second" are
     such arguments in the case of this function), which will come from
     the --redOpts and --blueOpts command-line arguments to capture.py.
     For the nightly contest, however, your team will be created without
     any extra arguments, so you should make sure that the default
     behavior is what you want for the nightly contest.
     """
     
     # The following line is an example only; feel free to change it.
     return [eval(first)(firstIndex), eval(second)(secondIndex)]
 
##########
# Agents #
##########
 
class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """
    
    depth = 3
    
    def registerInitialState(self, gameState):
        global originFoodList, originMine, myFoodEaten
        self.goingHome = False
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        origin = gameState
        originFoodList = self.getFood(gameState).asList()
        originMine = self.getFoodYouAreDefending(gameState).asList()
        myFoodEaten=0
         
    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)
        distance = []
        global foodCount
        
        print("")
        print("Starting New Round")
        print("")
        
        if gameState.getAgentPosition(self.index) == self.start:
            foodCount = False
        
        values = [self.evaluate(gameState, a) for a in actions]
    
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    
        foodList = self.getFood(gameState).asList()
        
        #Number of remaining pellets
        foodLeft = len(foodList)  
        
        #----------------------------------------------------------------------------------
        #Calculates how many pellets an offensive agent is holding
        #----------------------------------------------------------------------------------
        if(self.mode !='defense'):
            if self.red:
                if (gameState.getAgentPosition(self.index)[0] < gameState.getWalls().width/2):
                    self.myFoodEaten = 0
                    self.goingHome = False
                    foodCount = False
                else:
                    prev = self.getPreviousObservation()
                    if len(self.getFood(prev).asList()) != len(foodList):
                        self.myFoodEaten += len(self.getFood(prev).asList()) - len(foodList)
            else:
                if (gameState.getAgentPosition(self.index)[0] > gameState.getWalls().width/2):
                    self.myFoodEaten = 0
                    self.goingHome = False
                    foodCount = False
                else:
                    prev = self.getPreviousObservation()
                    if len(self.getFood(prev).asList()) != len(foodList):
                        self.myFoodEaten += len(self.getFood(prev).asList()) - len(foodList)
        else:
            self.myFoodEaten = 0
        otherTeam = self.getOpponents(gameState)
        otherPosition = [gameState.getAgentPosition(a) for a in otherTeam]
        
         
        #-------------------------------------------------------------------------------------------
        # Determines whether the agent should save its pellets by jumping back to the home side
        #-------------------------------------------------------------------------------------------
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            pos2 = successor.getAgentPosition(self.index)
            for food in foodList:       #Creates a list of the distance to each food
                distance.append(self.getMazeDistance(successor.getAgentPosition(self.index),food))
        closestFood = min(distance) #determines shortest distance to a pellet
        #In other words, if the dist to home is less than the closest pellet, and is agent has at least 5 pellets in his possession
        value = len(originFoodList)-foodLeft
        if (4.0/5.0)*manhattanDistance(gameState.getAgentPosition(self.index),(gameState.getWalls().width/2,gameState.getWalls().height/2)) < closestFood and self.myFoodEaten >= 5:
            foodCount = True
            
            
        #------------------------------------------------------------------                  
        # If activated, head for home side
        #------------------------------------------------------------------
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]   #Gets agentStates for opponents
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]  #Retains the enemies who are ghosts
        bestAction = Directions.STOP
        if foodLeft <= 2 or foodCount == True:
            bestDist = 9999
            self.goingHome = True
            for action in actions:
                badChoice = False
                successor = self.getSuccessor(gameState, action)    #Gets successor states
                pos2 = successor.getAgentPosition(self.index)
                for enemy in enemies:
                    if pos2 == self.start:      #Don't take action if it leads 
                       print("IS THIS ACTIV")
                       badChoice = True
                       break
                if badChoice or action == Directions.STOP:
                    print("active")
                    continue
                dist = self.getMazeDistance(self.start,pos2)    #distance between current position and successor position
                if dist < bestDist:     #Calculates minimum distance
                    print("HOW MANY TIMES")
                    bestAction = action
                    bestDist = dist
            print("BestAction", bestAction)
            return bestAction
        chosenOne = random.choice(bestActions)
        print("Random: ", chosenOne)
        return chosenOne 
    def getSuccessor(self, gameState, action):
      """
      Finds the next successor which is a grid position (location tuple).
      """
      successor = gameState.generateSuccessor(self.index, action)
      pos = successor.getAgentState(self.index).getPosition()
      if pos != nearestPoint(pos):
        # Only half a grid position was covered
        return successor.generateSuccessor(self.index, action)
      else:
        return successor
  
    def evaluate(self, gameState, action):
      """
      Computes a linear combination of features and feature weights
      """
      print("Action: ", action)
      features = self.getFeatures(gameState, action)
      weights = self.getWeights(gameState, action)
      print("Product: ", features*weights)
      return features * weights
    def getFeatures(self, gameState, action):
      """
      Returns a counter of features for the state
      """
      features = util.Counter()
      successor = self.getSuccessor(gameState, action)
      opponents = self.getOpponents()
      distances = [self.getMazeDistance(succesor.getAgentPosition(self.index), successor.getAgentPosition(ghost)) for ghost in opponents]
      features['successorScore'] = self.getScore(successor)
      features['ghostDistance'] = min(distances)
      return features
    def getWeights(self, gameState, action):
      """
      Normally, weights do not depend on the gamestate.  They can be either
      a counter or a dictionary.
      """
      return {'successorScore': 1.0, 'ghostDistance': -100.0}
  
class OffensiveAgent(DummyAgent):
   """
   A reflex agent that seeks food. This is an agent
   we give you to get an idea of what an offensive agent might look like,
   but it is by no means the best or only way to build an offensive agent.
   """
   
   #------------------
   #Initializes agent
   #------------------
   def registerInitialState(self, gameState):
        self.mode = 'offense'
        DummyAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
   
   #----------------------------------
   #Calculates features for the states
   #----------------------------------
   def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()   
    opponents = self.getOpponents(gameState)
    distances = successor.getAgentDistances()
    state = successor.getAgentState(self.index)
    position = state.getPosition()
    chase = False       #Tells the agent whether or not it is beign chased by an enemy
    
    for index in self.getTeam(successor):   #Finds the agent's teammate's index
            if index != self.index:
               teammate = index
               break
     
    features['successorScore'] = -len(foodList)
    
 
    # -----------------------------------------------------
    # Computes distance to defenders we can see
    # -----------------------------------------------------
    enemies = self.getOpponents(successor)   #Gets agentStates for opponents
    visible = [(a,successor.getAgentState(a)) for a in enemies if not successor.getAgentState(a).isPacman and successor.getAgentState(a).getPosition() != None]  #Retains the enemies who are ghosts
    if len(visible) > 0:
        enemy = visible[0]
    for element in visible:
        if self.getMazeDistance(successor.getAgentPosition(self.index), element[1].getPosition()) < self.getMazeDistance(successor.getAgentPosition(self.index), enemy[1].getPosition()):
            enemy=element
    #Activates if agent is on opponents side and is within range of enemy
    if len(visible) > 0 and state.isPacman:
            dist = abs(self.getMazeDistance(position, enemy[1].getPosition()))  #Find the enemy distance
            
            #Should activate if agent can eat the opponent
            if self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(enemy[0])) <= 1.5 and (enemy[1].getPosition() == None or successor.getAgentPosition(self.index) == gameState.getAgentPosition(enemy[0])):
                features['defenderDistance'] = 1000
                chase = False
            else: 
                #Pacman agent is within 4 tiles of enemy
                if dist <= 4.0:
                    if enemy[1].scaredTimer > 0 and enemy[1].scaredTimer != 40:
                        chase = False   #If timer is counting, agent is not being chased 
                    else:
                        chase = True        #If the timer is not activated, agent is being chased
                    if enemy[1].scaredTimer >= 2.0 and enemy[1].scaredTimer != 40:
                        features['defenderDistance'] = -1.0 * dist  #Minimize distance to enemy if they are scared
                    else:
                        features['defenderDistance'] = dist
                #Pacman agent is further that 4 tiles from an enemy
                else:
                    chase = False
                    features['defenderDistance'] = dist
                if successor.getAgentPosition(self.index) == self.start:
                    features['defenderDistance'] = -100
    #Agent must not by pacman but still within range of enemy
    elif len(visible) > 0:
        for enemy in visible:
            #Enemy is visible and within 4.0 units
            features['defenderDistance'] = self.getMazeDistance(successor.getAgentPosition(self.index), enemy[1].getPosition())
            if successor.getAgentPosition(self.index) == self.start:
                features['defenderDistance'] = -100
        if self.getMazeDistance(successor.getAgentPosition(teammate), enemy[1].getPosition()) <= 2.0 and self.getMazeDistance(successor.getAgentPosition(self.index), enemy[1].getPosition()) > 2.0 and len(visible) == 1: #Supposed to fix Mexican Standoff
            features['defenderDistance'] = 1
    #There are no visible enemies
    else: 
        features['defenderDistance'] = 10.0     #Arbitrary constant value
        if successor.getAgentPosition(self.index) != self.start:
            chase = False
    if features['defenderDistance'] > 6.0:      #Ignores enemies outside visibility range
        features['defenderDistance'] = 1.0
    #-----------------------------------------
    # Compute distance to the nearest food
    #---------------------------------------- 
    if len(foodList) > 0:
       myPos = successor.getAgentPosition(self.index)
       minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])  #Finds dist(successor,closestFood)
       originalDistance = min([self.getMazeDistance(gameState.getAgentPosition(self.index), food) for food in foodList]) #Finds dist(gameState,closestFood)
       #Activtes if the agent is still searching for pellets
       if not self.goingHome:
           #Should mean that successor contains a pellet
           if successor.getAgentPosition(self.index) in self.getFood(gameState).asList():
               features['distanceToFood'] = 10.0
           else:
               features['distanceToFood'] = 1.0/minDistance**2  #Prioritizes closer pellets
       else:
           features['distanceToFood'] = 0.0
           print("ACTIVATING BY ELSE")
       #Ignore food if chased
       if chase:
           print("ACTIVATING BY CHASE")
           if self.getMazeDistance(myPos,successor.getAgentPosition(teammate)) >= 2.0 and myPos == self.start:
               features['distanceToFood'] *= 0.0
     
    #--------------------------------------------
    # Computes Distance to nearest Power Pellet 
    #--------------------------------------------
    capsuleList = self.getCapsules(gameState)
    if len(capsuleList) > 0:
        myPos = successor.getAgentState(self.index).getPosition()
        minDistance = min([self.getMazeDistance(myPos, pellet) for pellet in capsuleList])
        if not self.goingHome:
            if minDistance == 0.0:      #Should mean that the state holds a power pellet
                features['distanceToPP'] = 1.0
            else:
                features['distanceToPP'] = 1.0/minDistance  #Prioritizes closer power pellets
        else:
            features['distanceToPP'] = 0.0
        if chase:
            features['distanceToPP'] *= 2.0     #Increases desire for power pellets if the agent is being chased
    else:
        features['distanceToPP'] = 0.0  #There are no power pellets remaining that the agent can use
    #------------------------------------
    # Determines if the path is a tunnel 
    #------------------------------------
    breakout = False        #Used to breakout of all of the loops from the inner loop
    if chase:   #Avoids tunnels if the agent is being chased
        nextActions = successor.getLegalActions(self.index)     #Actions possible  from the examined successor state
        if len(nextActions) <= 2.0:         #Return a really bad value if being chased and immediate successor is a dead end
            features['tunnel'] = -100
        else:
            for action in nextActions:  #Examine viable actions from successor state.  Loop should examine all states of depth 3 from current gameState
                if breakout:    #If the third state is a dead end, exit process
                    break
                nextInLine = self.getSuccessor(successor,action)    #Calculates successorState of successor
                #Skip examination of the STOP action and the action returning the agent to the original gameState
                if nextInLine.getAgentPosition(self.index) == gameState.getAgentPosition(self.index) or action == Directions.STOP:
                    continue
                followingNextActions = nextInLine.getLegalActions(self.index)
                if len(followingNextActions) <= 2.0:    #Return a bad value if the state represents a deadend (only legal actions either return to the previous state or are STOP)
                    features['tunnel'] = -75
                else:   #Repeats the above process for a third state
                    for nextAction in followingNextActions:
                        futureState = self.getSuccessor(nextInLine, nextAction)
                        if futureState.getAgentPosition(self.index) == successor.getAgentPosition(self.index) or nextAction == Directions.STOP:
                            continue
                        futureActions = futureState.getLegalActions(self.index)
                        if len(futureActions) <= 2.0:
                            features["tunnel"] = -50
                        else:   #Breaks the process if at one state is determined to not be a dead end
                            features["tunnel"] = 10
                            breakout = True
                            break
    else:
        features['tunnel'] = 10     #Arbitrary value in case the agent is not being chased 
     
    #--------------------------------------------------------------
    # Tells agent to keep moving
    #--------------------------------------------------------------
     
    if successor.getAgentState(self.index).getPosition() == gameState.getAgentState(self.index).getPosition():
        features['stopped'] = -10
    else:
        features['stopped'] = 10
    
    #------------------------------------------------------------------------
    # Don't backtrack while on home board
    #------------------------------------------------------------------------
    
    prev = self.getPreviousObservation()
    
    if prev is not None:
        if not prev.getAgentState(self.index).isPacman: #If the agent was pacman in the previous state...
            if successor.getAgentPosition(self.index) == prev.getAgentPosition(self.index) and len(visible) > 0:    #return a negative value if the agent has returned to the position of the previous state
                features['stopLookingBack']= -20.0
            else:
                features['stopLookingBack'] = 0.0
        else:
            features['stopLookingBack'] = 0.0   #Else functions in case the if statements above don't work
    else:
            features['stopLookingBack'] = 0.0
            
            
    print(features)
    return features
 
   #------------------------------------
   #Returns the weights for the features
   #------------------------------------
   def getWeights(self, gameState, action):
     return {'successorScore': 1.5, 'defenderDistance': 2.0, 'distanceToFood': 3.0, 'distanceToPP': 1.0, 'tunnel': 1.0, 'stopped': 1.0, 'stopLookingBack': 1.0}
  
class DefensiveAgent(DummyAgent):

  mode = 'defense'
  
  #------------------------------------------------
  #Initializes Agent
  #------------------------------------------------
  def registerInitialState(self, gameState):
    self.red = gameState.isOnRedTeam(self.index)
    self.distancer = distanceCalculator.Distancer(gameState.data.layout)
    self.distancer.getMazeDistances()
    self.getLegalPositions(gameState)
    self.initializeBeliefDistributions(gameState)
  
  #--------------------------------------------------------------------------
  #For sake of convenience, returns a list of all legal positions on the map
  #--------------------------------------------------------------------------
  def getLegalPositions(self, gameState):
    self.validPositions = []
    walls = gameState.getWalls()
    for x in range(walls.width):
        for y in range(walls.height):
            if not walls[x][y]:
                self.validPositions.append((x,y))

  #--------------------------------------------
  #Sets up the belief distributions per agent
  #--------------------------------------------
  def initializeBeliefDistributions(self, gameState):
    self.beliefDistributions = dict()
    for agent in self.getOpponents(gameState):
      self.initializeDistribution(gameState, agent)

  #--------------------------------------------------------------------------------
  #Initializes a uniform distribution around the board in which an enemy can reside
  #--------------------------------------------------------------------------------
  def initializeDistribution(self, gameState, agentIndex):
    self.beliefDistributions[agentIndex] = Counter()
    for (x,y) in self.validPositions:
      if (gameState.isOnRedTeam(agentIndex) and x <= gameState.getWalls().width/2) or (not gameState.isOnRedTeam(agentIndex) and x >= gameState.getWalls().width/2):  #Examines every grid square
        self.beliefDistributions[agentIndex][(x,y)] = 1.0   #Gives each grid square per agent and equal value
    self.beliefDistributions[agentIndex].normalize()    #Converts each value to a probability; each grid square has the same initial probability

  #------------------------------------------------------------
  #Chooses the best action for the agent to take
  #------------------------------------------------------------
  def chooseAction(self, gameState):
    currObservation = self.getCurrentObservation()
    self.observe(currObservation)   #Updates Probability Distributions
    self.deltaTime(currObservation)    #Attempts to add probabilistic predictions to belief distribution

    actions = currObservation.getLegalActions(self.index)
    myPos = currObservation.getAgentPosition(self.index)
    closestEnemy = self.getClosestEnemy(currObservation)
    bestAction = Directions.STOP
    
    enemyPos = currObservation.getAgentPosition(closestEnemy)
    if enemyPos is None: 
        enemyPos = self.beliefDistributions[closestEnemy].argMax()
    minDistance = self.getMazeDistance(myPos, enemyPos)
    for action in actions:
        successor = currObservation.generateSuccessor(self.index, action)
        myNewPos = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(myNewPos, enemyPos)
        if dist < minDistance and not successor.getAgentState(self.index).isPacman:
            minDistance = dist
            bestAction = action
    return bestAction

  #--------------------------------------
  #Updates the belief distribution
  #--------------------------------------
  def observe(self, observationState):
    myPos = observationState.getAgentPosition(self.index)
    noisyDistances = observationState.getAgentDistances()
    newDistribution = dict()
    
    for enemy in self.getOpponents(observationState):
        if self.beliefDistributions[enemy].totalCount() == 0:   #If the districution for an enemy is not set up, set it up
            self.initializeDistribution(observationState, enemy)
        distribution = Counter()
        if observationState.getAgentPosition(enemy) is not None:    #Assuming
            distribution[observationState.getAgentPosition(enemy)] = 1
        else:
            for position in self.validPositions:
                dist = manhattanDistance(myPos, position)
                distribution[position] = self.beliefDistributions[enemy][position] * observationState.getDistanceProb(dist, noisyDistances[enemy])
            distribution.normalize()    #Convert values to probabilities
        newDistribution[enemy] = distribution
    self.beliefDistributions = newDistribution

  #----------------------------------------------------------------
  #Tries to predict the likelihood of enemy positions in the future
  #----------------------------------------------------------------
  def deltaTime(self, observedState):
    newDistributions = dict()
    for agentIndex in self.getOpponents(observedState):
        distribution = Counter()
        for position in self.validPositions:
            newPositionDistribution = Counter()
            for neighboringPos in self.getLegalAdjPositions(observedState, position):
                newPositionDistribution[neighboringPos] = 1
            newPositionDistribution.normalize()
            for newPos, prob in newPositionDistribution.items():
                distribution[newPos] += self.beliefDistributions[agentIndex][position] * prob
        distribution.normalize()
        newDistributions[agentIndex] = distribution
    self.beliefDistributions = newDistributions

  #--------------------------------------------------
  #Find the closest enemy within the agent's sight
  #--------------------------------------------------
  def getClosestEnemy(self, observationState):
    myPos = observationState.getAgentPosition(self.index)
    closestEnemy = None
    isPacman = False
    minDistance = float('inf')

    for agent in self.getOpponents(observationState):
        enemyPos = observationState.getAgentPosition(agent)
        if enemyPos is None: 
            enemyPos = self.beliefDistributions[agent].argMax()
        enemyDist = self.getMazeDistance(myPos, enemyPos)
        if (not isPacman and (enemyDist < minDistance or observationState.getAgentState(agent).isPacman)) or (observationState.getAgentState(agent).isPacman and enemyDist < minDistance):
            if observationState.getAgentState(agent).isPacman: 
                isPacman = True
            minDistance = enemyDist
            closestEnemy = agent
          
    return closestEnemy

  #----------------------------------------------------------------
  #Return a list of adjacent grid squares that an agent can move to
  #----------------------------------------------------------------
  def getLegalAdjPositions(self, gameState, (x,y)):
    walls = gameState.getWalls()
    positions = [(x,y)]
    if x-1 >= 0 and not walls[x-1][y]: 
        positions.append((x-1,y))
    if y+1 < walls.height and not walls[x][y+1]: 
        positions.append((x,y+1))
    if x+1 < walls.width and not walls[x+1][y]: 
        positions.append((x+1,y))
    if y-1 >= 0 and not walls[x][y-1]: 
        positions.append((x,y-1))
    
    return positions
     
"""
Agent accomplishes nothing; purely for testing purposes
"""        
class DoNothingAgent(CaptureAgent):
     
    def chooseAction(self, gameState): 
        return Directions.STOP
