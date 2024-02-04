import random
from IPython.display import clear_output
import matplotlib.pylab as plt
import numpy as np

class Player:

    def show_board(self, board):
        print('|'.join(board[6:9]))
        print('|'.join(board[3:6]))
        print('|'.join(board[0:3]))

class HumanPlayer(Player):

    def reward(self, value, board):
        pass

    def make_move(self, board):
        while True:
            try:
                super().show_board(board)
                clear_output(wait=True)
                move = input('Your next move (cell index 1-9):')
                move = int(move)
                if not (move - 1 in range(9)) or board[move-1]!=' ' :
                    raise ValueError
            except ValueError:
                print('Invalid move; try again:\n')
            else:
                return move - 1

class AIPlayer(Player):

    def __init__(self, epsilon=0.4, alpha=0.3, gamma=0.9, default_q=1):
        self.EPSILON = epsilon
        self.ALPHA = alpha
        self.GAMMA = gamma
        self.DEFAULT_Q = default_q
        self.q = {}
        self.move = None
        self.prev_board = (' ',) * 9
        self.rewards = []

    def available_moves(self, board):
        return [i for i in range(9) if board[i] == ' ']

    def get_q(self, state, action):
        if self.q.get((state, action)) is None:
            self.q[(state, action)] = self.DEFAULT_Q
        return self.q[(state, action)]

    def make_move(self, board):
        self.prev_board = tuple(board)
        actions = self.available_moves(board)

        if random.random() < self.EPSILON:
            self.move = random.choice(actions)
        else:
            q_values = [self.get_q(self.prev_board, a) for a in actions]
            max_q_value = max(q_values)
            if q_values.count(max_q_value) > 1:
                best_actions = [i for i in range(len(actions)) if q_values[i] == max_q_value]
                best_move = actions[random.choice(best_actions)]
            else:
                best_move = actions[q_values.index(max_q_value)]
            self.move = best_move
        return self.move

    def reward(self, reward, board):
        prev_q = self.get_q(self.prev_board, self.move)
        max_q_new = max([self.get_q(tuple(board), a) for a in self.available_moves(self.prev_board)])
        self.q[(self.prev_board, self.move)] = prev_q + self.ALPHA * (reward + self.GAMMA * max_q_new - prev_q)
        self.rewards.append(reward)

class TicTacToe:

    BLANK = ' '
    AI_PLAYER = 'X'
    HUMAN_PLAYER = '0'
    REWARD_WIN = 10
    REWARD_LOSE = -10
    REWARD_TIE = 0

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.isHuman = isinstance(player1, HumanPlayer) or isinstance(player2, HumanPlayer)
        self.first_player_turn = random.choice([True, False])
        self.board = [' '] * 9

    def play(self):
        while True:
            if self.first_player_turn:
                player = self.player1
                other_player = self.player2
                player_markers = (self.AI_PLAYER, self.HUMAN_PLAYER)
            else:
                player = self.player2
                other_player = self.player1
                player_markers = (self.HUMAN_PLAYER, self.AI_PLAYER)

            game_over, winner = self.is_game_over(player_markers[1])

            if game_over:
                if winner:
                    player.reward(self.REWARD_LOSE, self.board[:])
                    other_player.reward(self.REWARD_WIN, self.board[:])
                    if self.isHuman:
                        other_player.show_board(self.board)
                        print('\n %s won!' % other_player.__class__.__name__)
                else:
                    player.reward(self.REWARD_TIE, self.board[:])
                    other_player.reward(self.REWARD_TIE, self.board[:])
                    if self.isHuman:
                        other_player.show_board(self.board)
                        print('Tie!')
                break
            self.first_player_turn = not self.first_player_turn
            move = player.make_move(self.board)        
            self.board[move] = player_markers[0]

    def is_game_over(self, player_marker):
        for i in range(3):
            if self.board[3 * i + 0] == player_marker and \
                    self.board[3 * i + 1] == player_marker and \
                    self.board[3 * i + 2] == player_marker:
                return True, player_marker
        for j in range(3):
            if self.board[j + 0] == player_marker and \
                    self.board[j + 3] == player_marker and \
                    self.board[j + 6] == player_marker:
                return True, player_marker
        if self.board[0] == player_marker and self.board[4] == player_marker and self.board[8] == player_marker:
            return True, player_marker
        if self.board[2] == player_marker and self.board[4] == player_marker and self.board[6] == player_marker:
            return True, player_marker

        if self.board.count(' ') == 0:
            return True, None
        else:
            return False, None

TRAINING_EXAMPLES = 10000000
TRAINING_EPSILON_1 = 0.2
TRAINING_EPSILON_2 = 0.3

ai_player_1 = AIPlayer()
ai_player_2 = AIPlayer()
print('Training the AI players...')
ai_player_1.EPSILON = TRAINING_EPSILON_1
ai_player_2.EPSILON = TRAINING_EPSILON_2

for _ in range(TRAINING_EXAMPLES):
    game = TicTacToe(ai_player_1, ai_player_2)
    game.play()

print('Training is Done')

# Plot training results
plt.subplots()
plt.plot(np.cumsum(ai_player_1.rewards))
plt.plot(np.cumsum(ai_player_2.rewards))
plt.xlabel('Training games')
plt.ylabel('Cumulative reward')
plt.title('Training progress')
plt.legend(['AI player 1','AI player 2'])

plt.subplots()
plt.plot(np.cumsum(ai_player_1.rewards[0:100]))
plt.plot(np.cumsum(ai_player_2.rewards[0:100]))
plt.xlabel('Training games')
plt.ylabel('Cumulative reward')
plt.title('Training progress for the first 100 games')
plt.legend(['AI player 1','AI player 2'])

plt.show()

# Play against the trained AI player
ai_player_1.EPSILON = 0
human_player = HumanPlayer()
game = TicTacToe(ai_player_1, human_player)
game.play()
