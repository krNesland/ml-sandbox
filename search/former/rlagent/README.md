Initial thoughts (these are a bit like Alpha Zero with Monte Carlo Tree Search):
- N random actions are taken
- The NN evaluates the N grids (each converted to a 1D array) that result from the actions
- Output of NN is a floating point number between 0 and 1 that scores how good a grid is
- The action that resulted in the grid with the highest score is chosen
- The action is then applied to the grid
- We then repeat the process (by a certain probability, we can choose to take a random action instead of the action that resulted in the best grid)
- Agent gets a reward equal to 1 / num_moves when it reaches the goal

Refined thoughts (more proper Deep Q-Learning):
- The NN takes the grid (converted to a 1D array) as input and outputs a score for each action (which is the same as for each cell in the grid)
- Depending on the exploration factor, we either choose the action with the highest score or a random action