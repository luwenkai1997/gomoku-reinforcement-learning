import os
import time
import random
import queue
import numpy as np
from collections import deque
import torch
import torch.optim as optim
import torch.nn.functional as F

from config import Config
from model import ResNetPolicyValueNet
from self_play import self_play_worker
from evaluate import evaluate_model

class TrainPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_buffer = deque(maxlen=cfg.buffer_size)
        
        self.policy_value_net = ResNetPolicyValueNet(board_size=cfg.board_size, 
                                                     n_res_blocks=cfg.n_res_blocks, 
                                                     n_filters=cfg.n_filters).to(cfg.device)
                                                     
        if cfg.n_gpus > 1:
            self.policy_value_net = torch.nn.DataParallel(self.policy_value_net)
            
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), 
                                    weight_decay=1e-4, lr=cfg.learn_rate)
                                    
        if cfg.restore_model and os.path.exists(cfg.restore_model):
            self.policy_value_net.load_state_dict(torch.load(cfg.restore_model, map_location=cfg.device))
            print(f"Loaded model from {cfg.restore_model}")
            
        self.best_win_ratio = 0.0
        self.lr_multiplier = cfg.lr_multiplier
        
    def policy_value_fn(self, board):
        state = board.current_state()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        
        with torch.no_grad():
            self.policy_value_net.eval()
            log_act_probs, value = self.policy_value_net(state_tensor)
            self.policy_value_net.train()
            
        act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
        legal_positions = list(board.availables)
        probs = [(move, act_probs[move]) for move in legal_positions]
        return probs, value.item()
        
    def collect_selfplay_data(self):
        q = queue.Queue() # Using thread-safe queue. Can be replaced with Multiprocessing queue for true parallelism
        for _ in range(self.cfg.games_per_iter):
            self_play_worker(self.cfg, self.policy_value_fn, q)
            winner, ep_len, play_data = q.get()
            self.data_buffer.extend(play_data)
            
    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.cfg.train_batch_size)
        state_batch = torch.tensor(np.array([data[0] for data in mini_batch]), dtype=torch.float32, device=self.cfg.device)
        mcts_probs_batch = torch.tensor(np.array([data[1] for data in mini_batch]), dtype=torch.float32, device=self.cfg.device)
        winner_batch = torch.tensor(np.array([data[2] for data in mini_batch]), dtype=torch.float32, device=self.cfg.device).unsqueeze(1)
        
        old_log_probs, _ = self.policy_value_net(state_batch)
        old_probs = torch.exp(old_log_probs).detach()
        
        for i in range(self.cfg.update_epochs):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.cfg.learn_rate * self.lr_multiplier
                
            self.optimizer.zero_grad()
            log_act_probs, value = self.policy_value_net(state_batch)
            
            value_loss = F.mse_loss(value, winner_batch)
            policy_loss = -torch.mean(torch.sum(mcts_probs_batch * log_act_probs, 1))
            loss = value_loss + policy_loss
            loss.backward()
            self.optimizer.step()
            
            with torch.no_grad():
                new_log_probs, _ = self.policy_value_net(state_batch)
                new_probs = torch.exp(new_log_probs)
                kl = torch.mean(torch.sum(old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10)), dim=1))
                if kl > self.cfg.kl_coeff * 4:
                    break
                    
        if kl > self.cfg.kl_coeff * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.cfg.kl_coeff / 2 and self.lr_multiplier < 10.0:
            self.lr_multiplier *= 1.5
            
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()
        
    def run(self):
        print(self.cfg.summary())
        for i in range(self.cfg.total_iterations):
            self.collect_selfplay_data()
            if len(self.data_buffer) > self.cfg.train_batch_size:
                loss, entropy = self.policy_update()
                print(f"Iter: {i}, Loss: {loss:.4f}, Entropy: {entropy:.4f}, lr_mult: {self.lr_multiplier:.4f}")
            else:
                print(f"Iter: {i}, filling buffer... ({len(self.data_buffer)}/{self.cfg.train_batch_size})")
                
            if (i+1) % self.cfg.check_freq == 0:
                model_path = os.path.join(self.cfg.checkpoint_dir, f"model_{i+1}.pt")
                torch.save(self.policy_value_net.state_dict(), model_path)
                print(f"Saved model to {model_path}")
                
                win_ratio = evaluate_model(self.cfg, self.policy_value_fn)
                print(f"Evaluation Win Ratio: {win_ratio:.2f}")
                if win_ratio > self.best_win_ratio:
                    self.best_win_ratio = win_ratio
                    torch.save(self.policy_value_net.state_dict(), os.path.join(self.cfg.checkpoint_dir, "best_model.pt"))
                    print("New best policy saved!")

if __name__ == '__main__':
    cfg = Config()
    pipeline = TrainPipeline(cfg)
    pipeline.run()
