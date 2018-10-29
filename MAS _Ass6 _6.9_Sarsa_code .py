
#Multi-agent systems homework 6 - Sarsa for example code

import numpy as np

MAX_COLUMN = 10
MAX_ROW = 5
CLIFF = [(4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8)]
TERMINAL = (4,9)


class Square:

    def __init__(self, row_coordinate,column_coordinate):
        self.row = row_coordinate
        self.column = column_coordinate
        self.position = (row_coordinate,column_coordinate)
        self.q_values = [0.0] * 4  #0 up, 1 for down, 2 for left, 3 for right for state action values

        if self.position in CLIFF or self.position == TERMINAL:
            self.up = (4,0)
            self.down = (4,0)
            self.left = (4,0)
            self.right = (4,0)
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

        if (final_row < 0 or final_column < 0 or final_row > 4 or final_column > 9):
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
        if (new_state in CLIFF):
            reward = -100
        elif (new_state == TERMINAL):
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
            if (index == 0):
                direction = "Up"
            elif (index == 1):
                direction = "down"
            elif (index == 2):
                direction = "Left"
            else: direction = "Right"
            print("Position: {} - Max = {}, Up Q: {},  Down Q: {}, left Q: {}, right Q: {}".format(converted_pos,direction,square.q_values[0], square.q_values[1], square.q_values[2],square.q_values[3]))

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
    alpha = 0.2
    converged = False
    grid = CreateGrid(MAX_ROW,MAX_COLUMN)
    while not converged:
        reached_terminal = False
        time_step = 1
        #epsilon = 1/time_step
        epsilon = 0.3
        initial_state = (4,0)
        initial_state_row = initial_state[0]
        initial_state_column = initial_state[1]
        #print("Initial state: {}".format(initial_state))
        print("Initial state: {}".format(tuple(map(lambda x: x+1,initial_state)))) #Plus 1 coordinates, to make it readable from 1 to 8
        current_square = grid[initial_state_row][initial_state_column]
        action = current_square.Choose_action(epsilon)

        while not reached_terminal:
            reward, new_state = current_square.take_action(action)
            lookahead_square = grid[new_state[0]][new_state[1]]
            lookahead_action = lookahead_square.Choose_action(epsilon)
            current_square.q_values[action] +=  alpha * (reward + GAMMA * lookahead_square.q_values[lookahead_action] - current_square.q_values[action])
            current_square = lookahead_square
            action = lookahead_action
            time_step += 1
            #epsilon = 1 / time_step
            #print("New current state is: {}".format(current_square.position))
            print("New current state is: {}".format(tuple(map(lambda x: x + 1, current_square.position))))  # Plus 1 coordinates, to make it readable from 1 to 8
            if (current_square.position == TERMINAL or current_square.position in CLIFF):
                reached_terminal = True
                converged = check_convergence(grid)
    print_grid(grid)

def main():
    sarsa_update()

main()







