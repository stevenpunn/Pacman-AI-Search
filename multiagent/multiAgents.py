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

        "*** YOUR CODE HERE ***"
        from util import manhattanDistance
        foodDistance = 0
        nextFoodDistance = 0     
        bonus = 0                            
        ghostPosition = successorGameState.getGhostPositions()
        nextFood = newFood.asList()                 # stores information about the next food node

        # if the gameState is won, return a large number
        if successorGameState.isWin():
          return 999999999999

        # find Manhattan Distance between pacman and ghost
        for ghost in ghostPosition:
          ghostDistance = manhattanDistance(newPos, ghost)        # calculates MH from pacman pos to current ghost position
          if ghostDistance <= 2:                                  # if ghost has a MH distance <= 2 to pacman, go in diff. direction
            return -999999999999
        
        # calculate score for pacman based on food location
        for food in nextFood:
          nextFoodDistance = manhattanDistance(newPos, food)    # finds MH between pacman and food position
          if not foodDistance:                                  # if foodDistance = 0, set = to MH of pacman and first food node
            foodDistance = nextFoodDistance
          if nextFoodDistance < foodDistance:                   # finds the shortest distance between the current food and previous food
            foodDistance = nextFoodDistance                     # stores nextFoodDistance as current node because it is the shortest distance

        if foodDistance <= 2:                                   # if food is extremely close to pacman, add 10 points
          bonus += 10
        elif foodDistance <= 5:                                 # if food is close but not close enough, add 5 points
          bonus += 5
        else:
          bonus += 1                                            # all other distances give add 1 point
        
        return successorGameState.getScore() + bonus            # returns current score + bonus for food proximity

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
        # Function to keep track of ghosts
        def minLevel(gameState, depth, numGhost):
            minVal = 99999999   # Intialize score to be large

            # Terminates if pacman wins or loses
            if gameState.isWin() or gameState.isLose() or depth == self.depth :
                return self.evaluationFunction(gameState)
            
            # Get list of legal actions for ghost
            ghostActions = gameState.getLegalActions(numGhost)

            # Recursively call minLevel to keep track of the score
            for actions in ghostActions:
                currGameState = gameState.generateSuccessor(numGhost, actions)
                numOfAgents = gameState.getNumAgents()  # Get total number of agents in state

                # Check if there is one last ghost remaining, if so, then it is the pacman, so call maxLevel
                if(numGhost == numOfAgents - 1):
                    minVal = min(minVal, maxLevel(currGameState, depth))  
                else:
                    minVal = min(minVal, minLevel(currGameState, depth, numGhost + 1))    # If not the last ghost, then add a ghost

            return minVal
        
        # Function to keep track of pacman
        def maxLevel(gameState, depth):
            maxVal = -99999999   # Intialize score to be small

            # Terminates if the pacman wins, loses, or reaches the maximum depth
            if gameState.isWin() or gameState.isLose() or (depth + 1) == self.depth:
                return self.evaluationFunction(gameState)
            
            # Get list of legal actions for pacman
            pacmanActions = gameState.getLegalActions(0)

            # Recursively call minLevel to keep track of the score and action
            for actions in pacmanActions:
                currGameState = gameState.generateSuccessor(0, actions)    
                maxVal = max(maxVal, minLevel(currGameState, depth + 1, 1))    # Update maximum value with pacman's best actions

            return maxVal
        
        # Pacman making decision at the root of the tree for best strategy
        actions = gameState.getLegalActions(0)
        score = -999999   # Intialize score to be small

        # Loop through each action of pacman and the minimum value score to calculate
        for a in actions:
            nextState = gameState.generateSuccessor(0, a)
            currScore = minLevel(nextState, 0, 1)   

            # Update score after each action is taken
            if currScore > score:
                score = currScore
                retAction = a

        return retAction    
            
        # util.raiseNotDefined()

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
    # accounts for randomness, includes failure
    #! Expectimax search: computer average score under optimal play
    #? what do we expect the average case outcome to be instead of best/worse
    # chance nodes instead of min nodes, but outcome is random
    # calculate expected utility: take weighed average of all options
    '''
      * if state = terminal: return state's utility
      * if next agent = MAX return max-value(state)
      * if next agent = EXP: return exp-value(state)  
    '''
    '''
      #? def max-value(state):
      #? init v = -infin
      #? for each succ of state: 
        #? v = max(v, val(succ))
      #? return v
    '''
    '''
      #* def exp-val(state):
      #* init v = 0
      #* for each succ of state:
        #* p = probabbility(succ)
        #* v += p * val*(succ)
      #* return v
    '''
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Function used for pacman agents
        def maxValue(gameState, depth):
            # Terminates if pacman wins, loses, or reaches maximum depth
            if gameState.isWin() or gameState.isLose() or (depth + 1) == self.depth:
                return self.evaluationFunction(gameState)
            
            actions = gameState.getLegalActions(0)  # Get list of actions
            maxVal = -99999999  # Set maximum value to be small 

            # Recursively call maxValue 
            for a in actions:
                successor = gameState.generateSuccessor(0, a)
                maxVal = max(maxVal, expectValue(successor, depth + 1, 1))  

            return maxVal
        
        # Function used for ghosts
        def expectValue(gameState, depth, agent):
            # Terminates if pacman wins or loses
            if gameState.isWin() or gameState.isLose():   
                return self.evaluationFunction(gameState)
            
            actions = gameState.getLegalActions(agent)  # Get list of actions
            numActions = len(actions)  # Get length of actions
            expectedValTotal = 0  # Set expected value total to 0

            # Recursively call maxValue or expectValue 
            for a in actions:
                successor = gameState.generateSuccessor(agent, a)
                if agent == (gameState.getNumAgents() - 1):
                    expectedVal = maxValue(successor, depth)  # Call maxValue
                else:
                    expectedVal = expectValue(successor, depth, agent + 1)  # Index of agent increased when there are still some ghosts and call expectValue
                expectedValTotal = expectedValTotal + expectedVal
                
            if numActions == 0:
                return  0
            
            return expectedValTotal / numActions
        
        # Pacman making deicision at the root of tree for best strategy
        actions = gameState.getLegalActions(0)
        currScore = -99999999   # Initialize score to be small

        # Loop through each action
        for a in actions:
            # Max is first with index of agent = 0
            successor = gameState.generateSuccessor(0, a)
            
            # Call expectValue because expect level is next with index of agent = 1, the first ghost
            score = expectValue(successor, 0, 1)

            # Action being chosen 
            if score > currScore:
                currScore = score
                retAction = a

        return retAction
        # util.raiseNotDefined()
      
def betterEvaluationFunction(currentGameState): 
  # want to evaluate states rather than actions, can use search code
  #! for reflex agents, may want to use reciprocal of important values (such as dist. to food) rather than the actual vals
  #! compute values for features about the state that are important, combine the features by multiplying & adding
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      # it was important to use the inverse of the distance to reward closer proximity
      # score was updated using inverse to encourage pacman to eat pellets closest
      # score is multiplied while ghost is closer to encourage avoiding ghosts
    """
    "*** YOUR CODE HERE ***"
    foodScore = currentGameState.getFood()
    foodList = foodScore.asList()
    pacmanPos = currentGameState.getPacmanPosition()
    updateScore = currentGameState.getScore()

    for food in foodList:
        distToFood = manhattanDistance(pacmanPos, food)       # calculates distance between pacman and nearest food
        updateScore += (1/distToFood)                         # food closest to pacman gets more points, use reciprocal
    
    ghostPos = currentGameState.getGhostStates()
    for ghost in ghostPos:
        findGhost = ghost.getPosition()
        distToGhost = manhattanDistance(pacmanPos, findGhost)    # finds position between pacman and ghost
        if (distToGhost) == 0:                                   # program doesn't work if ghost is too close, so need to add a condition
            continue
        else:
            updateScore += 5 * (1/distToGhost)                   # gives a higher score if the ghost is close to pacman, need to avoid
    return updateScore
    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

