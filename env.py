import numpy as np

class GomokuEnv:
    def __init__(self, board_size=15, n_in_row=5):
        self.board_size = board_size
        self.n_in_row = n_in_row
        self.players = [1, 2] # 1: Black, 2: White
        self.reset()
        
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        # Store moves explicitly as (player, move) to enable undo
        self.history = []
        self.current_player = self.players[0]
        self.availables = set(range(self.board_size * self.board_size))
        
    def do_move(self, move):
        if move not in self.availables:
            raise ValueError(f"Move {move} is not available.")
            
        h, w = move // self.board_size, move % self.board_size
        self.board[h, w] = self.current_player
        self.availables.remove(move)
        
        self.history.append((self.current_player, move))
        self.current_player = self.players[1] if self.current_player == self.players[0] else self.players[0]
        
    def undo_move(self):
        if not self.history:
            raise ValueError("No moves to undo.")
            
        prev_player, move = self.history.pop()
        h, w = move // self.board_size, move % self.board_size
        self.board[h, w] = 0
        self.availables.add(move)
        
        self.current_player = prev_player
        
    def current_state(self):
        """
        [4, board_size, board_size]
        0: current player pieces
        1: opponent pieces
        2: last move
        3: turn color
        """
        state = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)
        
        state[0] = (self.board == self.current_player).astype(np.float32)
        opponent = self.players[1] if self.current_player == self.players[0] else self.players[0]
        state[1] = (self.board == opponent).astype(np.float32)
        
        if self.history:
            last_move = self.history[-1][1]
            h, w = last_move // self.board_size, last_move % self.board_size
            state[2, h, w] = 1.0
            
        if len(self.history) % 2 == 0:
            state[3, :, :] = 1.0
            
        return state
        
    def has_winner(self):
        if len(self.history) < self.n_in_row * 2 - 1:
            return False, -1
            
        last_player, last_move = self.history[-1]
        h, w = last_move // self.board_size, last_move % self.board_size
        player_board = (self.board == last_player).astype(np.int8)
        
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dh, dw in directions:
            count = 1
            # Check forward
            r, c = h + dh, w + dw
            while 0 <= r < self.board_size and 0 <= c < self.board_size and player_board[r, c] == 1:
                count += 1
                r += dh
                c += dw
                
            # Check backward
            r, c = h - dh, w - dw
            while 0 <= r < self.board_size and 0 <= c < self.board_size and player_board[r, c] == 1:
                count += 1
                r -= dh
                c -= dw
                
            if count >= self.n_in_row:
                return True, last_player
                
        return False, -1
        
    def game_ended(self):
        win, winner = self.has_winner()
        if win:
            return True, winner
        elif len(self.availables) == 0:
            return True, -1 # Draw
        return False, -1
