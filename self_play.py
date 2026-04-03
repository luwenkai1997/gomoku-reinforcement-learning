import numpy as np

def get_equi_data(play_data, board_size):
    """augment the data set by rotation and flipping"""
    extend_data = []
    for state, mcts_prob, winner in play_data:
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(
                mcts_prob.reshape(board_size, board_size)), i)
            extend_data.append((equi_state,
                                np.flipud(equi_mcts_prob).flatten(),
                                winner))
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((equi_state,
                                np.flipud(equi_mcts_prob).flatten(),
                                winner))
    return extend_data

def self_play_worker(cfg, policy_value_fn, result_queue):
    from env import GomokuEnv
    from mcts import MCTSPlayer

    env = GomokuEnv(board_size=cfg.board_size, n_in_row=cfg.n_in_row)
    mcts_player = MCTSPlayer(policy_value_fn, c_puct=cfg.c_puct, 
                             n_playout=cfg.n_playout, is_selfplay=True,
                             noise_eps=cfg.noise_eps, dirichlet_alpha=cfg.dirichlet_alpha)
    
    states, mcts_probs, current_players = [], [], []
    env.reset()
    
    while True:
        temp = cfg.temperature if len(states) < cfg.temp_threshold else 1e-3
        move, move_probs = mcts_player.get_action(env, temp=temp, return_prob=True)
        
        states.append(env.current_state())
        mcts_probs.append(move_probs)
        current_players.append(env.current_player)
        
        env.do_move(move)
        
        end, winner = env.game_ended()
        if end:
            winners_z = np.zeros(len(current_players))
            if winner != -1:
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0
                
            mcts_player.reset_player()
            data = list(zip(states, mcts_probs, winners_z))
            augmented_data = get_equi_data(data, cfg.board_size)
            result_queue.put((winner, len(states), augmented_data))
            break
