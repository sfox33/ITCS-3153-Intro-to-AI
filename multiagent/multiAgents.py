# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #print("Position: ", newPos)
        #print("Food: ", newFood)
        #print("GhostStates: ", newGhostStates)
        #print("ScaredTimes: ", newScaredTimes)

        "*** YOUR CODE HERE ***"
        distance = []
        foodList = currentGameState.getFood().asList()  #Creates list of food positions
        pacmanPos = successorGameState.getPacmanPosition()    #Creates list of successor states for pacman
        newGhostPos = successorGameState.getGhostPositions()        #Retrives successor states for the ghosts
        ghostDist = [manhattanDistance(newPos, ghost) for ghost in newGhostPos] #Holds the distance to each ghost from the new state
    
        if(currentGameState.getPacmanPosition() == newPos):     #Encourages pacman to leave his current position
            return -10000
        
        for ghost in ghostDist:         #Returns an arbitrarily large negative value if a ghost is nearby
            if(ghost <= 1.0):
                return -10000
    
        for food in foodList:       #Creates a list of the distance to each food
            distance.append(-1.0*manhattanDistance(pacmanPos,food))
    
        return max(distance)
        
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        currentAgent = 0    #Holds the index for the agent being examined
        currentDepth = 0    #Holds the current depth of the tree being examined
        val = self.value(gameState, currentAgent, currentDepth) #Determines the value of the current state
        return val[0]  
    
    """
        Determines the value of a state.
    """
    def value(self, gameState, agent, currDepth):
        if (agent >= gameState.getNumAgents()): #If all agents have been examined, increase the depth and reset the agent count
            agent = 0
            currDepth += 1
        
        if (currDepth == self.depth):       #If the end of the depth is reached, evaluate the state
            return self.evaluationFunction(gameState)
        
        if (agent == 0):        #If the examined agent is pacman, program is at a max node
            return self.maxNode(gameState, agent, currDepth)
        else:       #If the examined state is not pacman (a ghost), program is at a min node.
            return self.minNode(gameState, agent, currDepth)
    
    """
        Finds the value of a Min node
    """        
    def minNode(self, gameState, agent, currDepth):
        val = ("", float("inf"))    #Sets the value to be an arbitrary number
        
        if len(gameState.getLegalActions(agent)) == 0:  #If there are no legal moves, evaluate the state
            return self.evaluationFunction(gameState)
                
        for action in gameState.getLegalActions(agent):
            childVal = self.value(gameState.generateSuccessor(agent, action), agent + 1, currDepth)     #Retrieve a successor state
            if (type(childVal) is tuple):   #Confirms that the childVal is the value of the state
                childVal = childVal[1] 

            newVal = min(val[1], childVal)  #Determines the smallest value of the examined nodes

            if (newVal is not val[1]):  #Confirms that the val variable holds the smallest valued node
                val = (action, newVal) 
        
        return val
    
    """
        Finds the value of a Max node
    """
    def maxNode(self, gameState, agent, currDepth):  
        val = ("", -1*float("inf"))     #Sets the value to be an arbitrary number
        
        if len(gameState.getLegalActions(agent)) == 0:  #If there are no legal moves, evaluate the state
            return self.evaluationFunction(gameState)
                
        for action in gameState.getLegalActions(agent):
            childVal = self.value(gameState.generateSuccessor(agent, action), agent + 1, currDepth)     #Retrieve a successor state
            if (type(childVal) is tuple):   #Confirms that the childVal is the value of the state
                childVal = childVal[1] 

            newVal = max(val[1], childVal)  #Determines the largest value of the examined nodes

            if (newVal is not val[1]):  #Confirms that the val variable holds the smallest valued node
                val = (action, newVal) 
        
        return val
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        currentAgent = 0    #Holds the index for the agent being examined
        currentDepth = 0    #Holds the current depth of the tree being examined
        val = self.value(gameState, currentAgent, currentDepth) #Determines the value of the current state
        return val[0]   
    
    """
        Determines the value of a state.
    """
    def value(self, gameState, agent, currDepth):
        if (agent >= gameState.getNumAgents()): #If all agents have been examined, reset the agent count and increase the depth
            agent = 0
            currDepth += 1
        
        if (currDepth == self.depth):   #If the depth has been reached, examine the current state
            return self.evaluationFunction(gameState)
        
        if (agent == 0):    #IF agent is pacman, the program is at a Max node
            return self.maxNode(gameState, agent, currDepth)
        else:       #If the agent is not pacman, the program is at an Expected Value node
            return self.expNode(gameState, agent, currDepth)
    
    """
        Finds the value of a Max node
    """        
    def maxNode(self, gameState, agent, currDepth):  
        val = ("", -1*float("inf"))     #Sets the value to an arbitrary number
        
        if len(gameState.getLegalActions(agent)) == 0:      #If there are no legal actions, evaluate the state
            return self.evaluationFunction(gameState)
                
        for action in gameState.getLegalActions(agent):
            childVal = self.value(gameState.generateSuccessor(agent, action), agent + 1, currDepth) #Retrieve a successor state
            if (type(childVal) is tuple):   #Confirms that the childVal is the value of the state
                childVal = childVal[1] 

            newVal = max(val[1], childVal)  #Determines the largest value of the examined children

            if (newVal is not val[1]):  #Confirms that the val variable is the largest-values child
                val = (action, newVal) 
        
        return val
    
    """
        Finds the value of an Expected Value node
    """
    def expNode(self, gameState, agent, currDepth):
        val = 0 #Sets the value to be 0
        
        if len(gameState.getLegalActions(agent)) == 0:  #If there are no legal actions, examine the state
            return self.evaluationFunction(gameState)
        
        actions = gameState.getLegalActions(agent)
        
        for action in gameState.getLegalActions(agent):       
            probability = 1.0/float(len(actions))       #Determine probability of state
            childVal = self.value(gameState.generateSuccessor(agent, action), agent + 1, currDepth) #Retrieve successor state
            
            if (type(childVal) is tuple):   #Confirms that the childVal is the value of the state
                childVal = childVal[1]
            
            val += probability * childVal   #Determines the child's portion of the expected value of the state
        
        return val
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

