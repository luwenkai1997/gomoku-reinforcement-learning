import numpy as np
from env import GomokuEnv
from mcts import MCTSPlayer

def pure_policy_value_fn(board):
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0

def evaluate_model(cfg, policy_value_fn):
    env = GomokuEnv(board_size=cfg.board_size, n_in_row=cfg.n_in_row)
    current_mcts_player = MCTSPlayer(policy_value_fn, c_puct=cfg.c_puct, n_playout=cfg.n_playout_eval)
    pure_mcts_player = MCTSPlayer(pure_policy_value_fn, c_puct=5, n_playout=cfg.pure_mcts_playout)
    
    win_cnt = {1: 0, -1: 0, 0: 0} # 1: current wins, -1: current loses, 0: tie
    
    for i in range(cfg.n_eval_games):
        env.reset()
        players = {1: current_mcts_player, 2: pure_mcts_player} if i % 2 == 0 else {1: pure_mcts_player, 2: current_mcts_player}
        
        while True:
            player_in_turn = players[env.current_player]
            move = player_in_turn.get_action(env)
            env.do_move(move)
            
            end, winner = env.game_ended()
            if end:
                if winner == -1:
                    win_cnt[0] += 1
                elif players[winner] == current_mcts_player:
                    win_cnt[1] += 1
                else:
                    win_cnt[-1] += 1
                    
                current_mcts_player.reset_player()
                pure_mcts_player.reset_player()
                break
                
    win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[0]) / cfg.n_eval_games
    print(f"Eval output: Win: {win_cnt[1]}, Lose: {win_cnt[-1]}, Tie: {win_cnt[0]}")
    return win_ratio
