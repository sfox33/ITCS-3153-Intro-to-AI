# search.py
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

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sets
import searchAgents
from pacman import GameState

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    "*** YOUR CODE HERE ***"
    root = [problem.getStartState()]    #Holds the root/initial state
    visited = sets.Set()    #Holds all nodes that have been examined
    parents = {}    #Dictionary that retains a tree structure by linking children to their parents
    frontier = util.Stack() #Holds the nodes currently being examined
    path_q = util.Queue()     #Once the DFS is finished, path holds the solution
    path_l = []
    frontier.push(root)
    while(not frontier.isEmpty()):
        v=frontier.pop()
        if(problem.isGoalState(v[0])):      #If the examined state is a goal state, end the DFS
            visited.add(v[0])
            curr = v
            while(root[0]!=curr[0]):    #Builds the path from the initial state to the goal state
                path_q.push(curr[1])
                nextCurr=parents[curr]
                curr=nextCurr
            while(not path_q.isEmpty()):        #Converts the frontier to a list
                path_l.append(path_q.pop())
            while(not frontier.isEmpty()):      #Empties the frontier to exit the loop
                frontier.pop()
        if(v[0] not in visited):        #Activate if the state has not been examined yet
            visited.add(v[0])
            for w in problem.getSuccessors(v[0]):       #Check each successor state(child)
                if(w[0] not in visited):
                    frontier.push(w)
                    parents[w]=v  
    path_l.reverse()
    return path_l

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    root = problem.getStartState()    #Holds the root of the tree.  The empty list is a spot holder to mimic the return value of the getSuccessors() method
    visited = []    #Holds all nodes that have been examined
    frontier = util.Queue() #Holds the nodes currently being examined
    frontier.push((root, []))     #Creates a priority queue to hold the states; empty list holds the actions used to get to the examined state
    while(not frontier.isEmpty()):
        v,actions=frontier.pop()    #The state(node) being examined
        if(problem.isGoalState(v)):       #End BFS if goal state is found
            return actions
        if(v not in visited):     #Activate if state has not been examined yet
            visited.append(v)
            for coordinates, action, cost in problem.getSuccessors(v):    #Examines each successor(child) of the examined state
                newActions = actions+[action]   #Calculates the cost of moving to this state
                frontier.push((coordinates,newActions))
    notFound()      #Returns an 'error' if a goal state was not found


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    root = [problem.getStartState(), []]    #Holds the root of the tree.  The empty list is a spot holder to mimic the return value of the getSuccessors() method
    visited = sets.Set()    #Holds all nodes that have been examined
    frontier = util.PriorityQueue() #Holds the nodes currently being examined
    frontier.push((root, []),0)     #Creates a priority queue to hold the states; empty list holds the actions used to get to the examined state
    while(not frontier.isEmpty()):
        v=frontier.pop()    #The state(node) being examined
        actions=v[1]    #A list of actions taken to get pacman to this state
        if(problem.isGoalState(v[0][0])):       #End UFC if goal state is found
            return v[1]
        if(v[0][0] not in visited):     #Activate if state has not been examined yet
            visited.add(v[0][0])
            for w in problem.getSuccessors(v[0][0]):    #Examines each successor(child) of the examined state
                if(w[0] not in visited):
                    newActions = actions+[w[1]]     #Calculates the cost of moving to this state
                    frontier.push((w,newActions), problem.getCostOfActions(newActions))
    notFound()      #Returns an 'error' if a goal state was not found

def notFound():
    print("Goal State Not Found")

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    root = problem.getStartState()   #Holds the root of the tree.  The empty list is a spot holder to mimic the return value of the getSuccessors() method
    visited = []    #Holds all nodes that have been examined
    frontier = util.PriorityQueue() #Holds the nodes currently being examined
    frontier.push((root, []), heuristic(root, problem))     #Creates a priority queue to hold the states; empty list holds the actions used to get to the examined state
    while(not frontier.isEmpty()):
        v,actions = frontier.pop()    #The state(node) being examined
        if(problem.isGoalState(v)):       #End A* if goal state is found
            return actions
        if(v not in visited):     #Activate if state has not been examined yet
            visited.append(v)
            for coordinates, action, cost in problem.getSuccessors(v):    #Examines each successor(child) of the examined state
                newActions = actions+[action]     #Calculates the cost of moving to this state
                frontier.push((coordinates,newActions), problem.getCostOfActions(newActions)+heuristic(coordinates, problem))
    notFound()      #Returns an 'error' if a goal state was not found

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
