import torch
import numpy as np
from config import Config
from model import ResNetPolicyValueNet
from mcts import MCTSPlayer
from env import GomokuEnv

def human_vs_ai(model_path, is_human_black=True):
    cfg = Config()
    
    # Load model
    model = ResNetPolicyValueNet(board_size=cfg.board_size, 
                                 n_res_blocks=cfg.n_res_blocks, 
                                 n_filters=cfg.n_filters)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    def policy_value_fn(board):
        state = board.current_state()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            log_act_probs, value = model(state_tensor)
        act_probs = np.exp(log_act_probs.numpy().flatten())
        legal_positions = list(board.availables)
        probs = [(move, act_probs[move]) for move in legal_positions]
        return probs, value.item()
        
    env = GomokuEnv(board_size=cfg.board_size, n_in_row=cfg.n_in_row)
    mcts_player = MCTSPlayer(policy_value_fn, c_puct=cfg.c_puct, n_playout=cfg.n_playout_eval)
    
    human_color = 1 if is_human_black else 2
    ai_color = 2 if is_human_black else 1
    
    while True:
        # Visualize
        print("\nBoard status:")
        for r in range(cfg.board_size):
            row_str = f"{r:2d} "
            for c in range(cfg.board_size):
                piece = env.board[r, c]
                if piece == 1: row_str += "X " # Black
                elif piece == 2: row_str += "O " # White
                else: row_str += ". "
            print(row_str)
        print("   " + " ".join([str(c%10) for c in range(cfg.board_size)]))
        
        end, winner = env.game_ended()
        if end:
            if winner == -1: print("Game ended in a tie!")
            elif winner == human_color: print("You won!")
            else: print("AI won!")
            break
            
        if env.current_player == human_color:
            while True:
                try:
                    move_str = input(f"Your move (format 'row col', e.g., '7 7'): ")
                    r, c = map(int, move_str.strip().split())
                    move = r * cfg.board_size + c
                    if move in env.availables:
                        env.do_move(move)
                        mcts_player.mcts.update_with_move(move)
                        break
                    else:
                        print("Invalid move, spot already taken or out of bounds.")
                except Exception as e:
                    print("Invalid input.")
        else:
            print("AI is thinking...")
            move = mcts_player.get_action(env, temp=1e-3)
            env.do_move(move)
            print(f"AI plays at: {move // cfg.board_size} {move % cfg.board_size}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python play.py <path_to_model.pt> [human_first (1/0)]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    human_first = True
    if len(sys.argv) > 2:
        human_first = bool(int(sys.argv[2]))
        
    human_vs_ai(model_path, human_first)
