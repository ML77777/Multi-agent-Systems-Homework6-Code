
#Multi-agent systems homework 6 - Sarsa for example code

import numpy as np

MAX_ROW_COLUMN = 8
WALLS = [(6, 1), (6, 2), (6, 3), (4, 5), (3, 5), (2, 5), (1, 5), (1, 2), (1, 3), (1, 4)]
TERMINALS = [(5, 4), (7, 7)]


class Square:

    def __init__(self, row_coordinate,column_coordinate):
        self.row = row_coordinate
        self.column = column_coordinate
        self.position = (row_coordinate,column_coordinate)
        self.q_values = [0.0] * 4  #0 up, 1 for down, 2 for left, 3 for right for state action values

        if self.position in WALLS or self.position in TERMINALS:
            self.up = self.position
            self.down = self.position
            self.left = self.position
            self.right = self.position
        else:
            self.up = self.MoveAndCheck(-1,0)
            self.down = self.MoveAndCheck(1,0)
            self.left = self.MoveAndCheck(0,-1)
            self.right = self.MoveAndCheck(0,1)

        self.actions_positions = [self.up,self.down,self.left,self.right]
        self.amount_action_taken = [0,0,0,0]

    def MoveAndCheck(self,move_row,move_column):
        final_row = self.row + move_row
        final_column = self.column + move_column
        final_position = (final_row,final_column)

        if (final_position in WALLS):
            final_row = self.row
            final_column = self.column
        elif (final_row < 0 or final_column < 0 or final_row > 7 or final_column > 7):
            final_row = self.row
            final_column = self.column

        return (final_row,final_column)

    def Choose_action(self, epsilon):
        #Find the best q values that this square has and divide its probabilities. 
        max_q_value = max(self.q_values)
        max_amount = self.q_values.count(max_q_value)
        amount_max_found = 0
        max_indices = []
        action_probabilities = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            if self.q_values[i] == max_q_value:
                amount_max_found += 1
                max_indices.append(i)
                action_probabilities[i] = 1 / max_amount

        epsilon_greedy_outcome = np.random.choice(2, 1, replace=False, p=[epsilon,1-epsilon])[0]
        if epsilon_greedy_outcome == 0:
            action = np.random.choice(4, 1, replace=False)[0]
        else:
            #print(max_indices)
            action = np.random.choice(max_indices, 1, replace=False)[0]

        return action

    def take_action(self,action):

        new_state = self.actions_positions[action]
        if (self.position in TERMINALS):
            reward = 0
        elif (new_state == TERMINALS[0]):
            reward = -20
        elif (new_state == TERMINALS[1]):
            reward = 10
        else:
            reward = -1

        reward_new_state = [reward, new_state]
        self.amount_action_taken[action] += 1
        return reward_new_state


def CreateGrid(row_size,column_size):
    grid = []

    for row  in range (row_size):
        row_squares = []
        for column in range(column_size):
            square = Square(row,column)
            row_squares.append(square)
        grid.append(row_squares)

    return grid

def print_grid(grid):

    for row_list in grid:
        for square in row_list:

            converted_pos = tuple(map(lambda x: x + 1, square.position)) # Plus 1 coordinates, to make it readable from 1 to 8
            max_q = max(square.q_values)
            index = square.q_values.index(max_q)
            print("Position: {} - Up Q: {},  Down Q: {}, left Q: {}, right Q: {}, Max = {}".format(converted_pos,square.q_values[0], square.q_values[1], square.q_values[2],square.q_values[3],index))

def check_convergence(grid):
    GAMMA = 1
    for row_list in grid:
        for current_square in row_list:
            for action in range(4):
                reward, new_state = current_square.take_action(action)
                alpha = 1 / current_square.amount_action_taken[action]
                lookahead_square = grid[new_state[0]][new_state[1]]
                for lookahead_action in range(4):
                    old = current_square.q_values[action]
                    new_q_value = current_square.q_values[action] + alpha * (reward + GAMMA * lookahead_square.q_values[lookahead_action] - current_square.q_values[action])
                    if ( (new_q_value - old) > 0.01):
                        return False
    return True


def sarsa_update():
    GAMMA = 1.0
    alpha = 0.5
    converged = False

    grid = CreateGrid(MAX_ROW_COLUMN,MAX_ROW_COLUMN)
    while not converged:

        reached_terminal = False
        initial_state = tuple(np.random.choice(8, 2, replace=True))
        while initial_state in WALLS or initial_state in TERMINALS:
            initial_state = tuple(np.random.choice(8, 2, replace=True))
        initial_state_row = initial_state[0]
        initial_state_column = initial_state[1]
        #print("Initial state: {}".format(initial_state))
        print("Initial state: {}".format(tuple(map(lambda x: x+1,initial_state)))) #Plus 1 coordinates, to make it readable from 1 to 8
        current_square = grid[initial_state_row][initial_state_column]
        #epsilon = 1/amount_episodes
        time_step = 1
        epsilon = 1/time_step
        action = current_square.Choose_action(epsilon)
        while not reached_terminal:

            reward, new_state = current_square.take_action(action)
            lookahead_square = grid[new_state[0]][new_state[1]]
            lookahead_action = lookahead_square.Choose_action(epsilon)
            #alpha = 1/(current_square.amount_action_taken[action]) 
            current_square.q_values[action] +=  alpha* (reward + GAMMA * lookahead_square.q_values[lookahead_action] - current_square.q_values[action])
            current_square = lookahead_square
            action = lookahead_action
            time_step += 1
            epsilon = 1 / time_step
            #print("New current state is: {}".format(current_square.position))
            print("New current state is: {}".format(tuple(map(lambda x: x + 1, current_square.position))))  # Plus 1 coordinates, to make it readable from 1 to 8
            if (current_square.position in TERMINALS):
                reached_terminal = True
                converged = check_convergence(grid)
    print_grid(grid)


def main():
    sarsa_update()

main()







