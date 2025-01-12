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
#import util.convert

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

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # 1. Loop through the nodes, through the parent nodes then to the child nodes
    # 2. If the node is visited, mark it as visited, and store visited nodes
    # 3. If the node is not visited, expand onto the child nodes
    # 4. If the child nodes haven't been visited, push onto stack

    # frontier = discovered but not explored
    # push = add element to the top of the stack
    # pop = remove and return item on top of stack
    # need to consider cost and direction
    '''
    For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor'
            '''

    # util.raiseNotDefined()
    frontier = util.Stack()                 # not final list, use to pop from
    visitedNodes = []                       # stores nodes to not visit again
    moves = []                              # this will be the final moves list
    currNode = []
    startSearch = problem.getStartState()
    startNode = (startSearch, [])           # need to keep [] to insert a new list
    frontier.push(startNode)                # insert start into frontier

    while not frontier.isEmpty():                                   # loops while there are nodes to check
        currNode, moves = frontier.pop()                            # need to keep together
        if problem.isGoalState(currNode):                           # end loops if current node is the goal state
            return moves
        if currNode not in visitedNodes:                            
            visitedNodes.append(currNode)                           # add current node to visitedNodes
            successors = problem.getSuccessors(currNode)            # use to import successor states from util
            for successor, action, cost in successors:          
                    updateAction = moves + [action]                 # creates new actions list, adds action to moves list
                    updateNode = (successor, updateAction)          # creates a list that stores next node to explore, and next move
                    frontier.push(updateNode)                       
    return moves


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    frontier = util.Queue()                 # same as DFS, just change to a queue
    visitedNodes = []                       # stores nodes to not visit again
    moves = []                              # this will be the final moves list
    currNode = []
    startSearch = problem.getStartState()
    startNode = (startSearch, [])           
    frontier.push(startNode)                # insert start into frontier

    while not frontier.isEmpty():                                   # loops while there are nodes to check
        currNode, moves = frontier.pop()                            # need to keep together
        if problem.isGoalState(currNode):                           # end loops if goal is reached
            return moves
        if currNode not in visitedNodes:                            
            visitedNodes.append(currNode)
            successors = problem.getSuccessors(currNode)            # use to import successor states from util
            for successor, action, stepCost in successors:          # keep stepCost or doesnt work
                    updateAction = moves + [action]                 # creates new actions list, adds action to moves list
                    updateNode = (successor, updateAction)          # creates a list that stores next node to explore, and next move
                    frontier.push(updateNode)                       
    return moves

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    frontier = util.PriorityQueue()         # same as DFS and BFS, just use a priorityQueue
    visitedNodes = []                       # stores nodes to not visit again
    moves = []                              # this will be the final moves list
    currNode = []
    startSearch = problem.getStartState()
    startNode = (startSearch, [])           # need to keep [] to insert a new list
    frontier.push(startNode, 0)                # insert start into frontier, also needs the priority (0)

    while not frontier.isEmpty():                                   # loops while there are nodes to check
        currNode, moves = frontier.pop()                            # need to keep together or doesnt work
        if problem.isGoalState(currNode):                           # end loops if goal is reached
            return moves
        if currNode not in visitedNodes:                            
            visitedNodes.append(currNode)
            successors = problem.getSuccessors(currNode)            # use to import successor states from util
            for successor, action, stepCost in successors:          # keep stepCost or doesnt work
                    updateAction = moves + [action]                 # creates new actions list, adds action to moves list
                    updateNode = (successor, updateAction)          # creates a list that stores next node to explore, and next move
                    cost = problem.getCostOfActions(updateAction)
                    if successor not in visitedNodes:
                         updateNode = (successor, updateAction)
                         frontier.push(updateNode, cost)             # push the new updated nodem uses cost to determine priority
    return moves

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # A* = cost of path + heuristic 
    frontier = util.PriorityQueue()                 # not final list, use to pop from
    visitedNodes = []                       # stores nodes to not visit again
    moves = []                              # this will be the final moves list
    currNode = []
    startSearch = problem.getStartState()
    startNode = (startSearch, [])           # need to keep [] to insert a new list
    frontier.push(startNode, 0)                # insert start into frontier, also needs the priority (0)

    while not frontier.isEmpty():                                   # loops while there are nodes to check
        currNode, moves = frontier.pop()                            # need to keep together or doesnt work
        if problem.isGoalState(currNode):                           # end loops if goal is reached
            return moves
        if currNode not in visitedNodes:                            
            visitedNodes.append(currNode)
            successors = problem.getSuccessors(currNode)            # use to import successor states from util
            for successor, action, stepCost in successors:          # keep stepCost or doesnt work
                    updateAction = moves + [action]                 # creates new actions list, adds action to moves list
                    cost = problem.getCostOfActions(updateAction) + heuristic(successor, problem)
                    if successor not in visitedNodes:
                        updateNode = (successor, updateAction)          # creates a list that stores next node to explore, and next move
                        frontier.push(updateNode, cost)             # push the new updated node, as well as the cost
    return moves


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
