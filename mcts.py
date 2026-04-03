import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode:
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # action: TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        
    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
                
    def select(self, c_puct):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))
        
    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits
        
    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)
        
    def get_value(self, c_puct):
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u
        
    def is_leaf(self):
        return self._children == {}
        
    def is_root(self):
        return self._parent is None

class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        
    def _playout(self, state):
        node = self._root
        moves_done = []
        while(1):
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)
            moves_done.append(action)
            
        end, winner = state.game_end() if hasattr(state, 'game_end') else state.game_ended()
        if not end:
            action_probs, leaf_value = self._policy(state)
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.current_player else -1.0
                
        node.update_recursive(-leaf_value)
        
        # Undo all moves to restore state
        for _ in reversed(moves_done):
            state.undo_move()
            
    def get_move_probs(self, state, temp=1e-3, add_noise=False, noise_eps=0.25, dirichlet_alpha=0.3):
        for n in range(self._n_playout):
            self._playout(state)
            
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        
        if temp < 1e-3: # greedy
            act_probs = np.zeros(len(visits))
            act_probs[np.argmax(visits)] = 1.0
        else:
            act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
            
        if add_noise:
            act_probs = (1 - noise_eps) * act_probs + noise_eps * np.random.dirichlet(dirichlet_alpha * np.ones(len(act_probs)))
            
        return acts, act_probs
        
    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

class MCTSPlayer:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=400, is_selfplay=False, 
                 noise_eps=0.25, dirichlet_alpha=0.3):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.noise_eps = noise_eps
        self.dirichlet_alpha = dirichlet_alpha
        
    def set_player_ind(self, p):
        self.player = p
        
    def reset_player(self):
        self.mcts.update_with_move(-1)
        
    def get_action(self, board, temp=1e-3, return_prob=False):
        sensible_moves = list(board.availables)
        move_probs = np.zeros(board.board_size * board.board_size)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(
                board, temp, add_noise=self._is_selfplay, 
                noise_eps=self.noise_eps, dirichlet_alpha=self.dirichlet_alpha)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)
            
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")
            return None
