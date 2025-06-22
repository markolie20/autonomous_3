import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
import random
import cv2 # opencv-python-headless

# Project-specific imports for the network and replay buffer
from ..networks.dqn_cnn import DQN_CNN
from ..utils.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, player_name, input_shape_raw, num_actions,
                 buffer_size=10000, batch_size=32, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=50000,
                 learning_rate=0.0001, target_update_frequency=1000,
                 learn_start_steps=1000, frame_stack_size=4,
                 img_height=84, img_width=84):
        self.name = player_name
        self.num_actions = num_actions
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.learning_rate = learning_rate
        self.target_update_frequency = target_update_frequency # In agent steps
        self.learn_start_steps = learn_start_steps # In agent steps

        self.frame_stack_size = frame_stack_size
        self.img_height = img_height
        self.img_width = img_width
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQNAgent '{self.name}' using device: {self.device}")

        # Use the imported DQN_CNN
        self.policy_net = DQN_CNN(frame_stack_size, num_actions, img_height, img_width).to(self.device)
        self.target_net = DQN_CNN(frame_stack_size, num_actions, img_height, img_width).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is not trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # Use the imported ReplayBuffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        self._frame_stack = collections.deque(maxlen=frame_stack_size)
        self.new_episode_started = True
        
        self.current_epsilon = eps_start
        self.total_steps_taken = 0 # Agent's own steps

    def _preprocess_frame(self, frame):
        if frame is None:
            return np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
        return frame

    def get_stacked_frames(self, raw_observation):
        processed_frame = self._preprocess_frame(raw_observation)
        
        if self.new_episode_started:
            self._frame_stack.clear()
            for _ in range(self.frame_stack_size):
                self._frame_stack.append(processed_frame)
            self.new_episode_started = False
        else:
            self._frame_stack.append(processed_frame)
        
        stacked_np_array = np.array(list(self._frame_stack), dtype=np.float32) / 255.0
        return stacked_np_array

    def reset_episode_state(self):
        self.new_episode_started = True

    def act(self, stacked_frames_state, training=True):
        self.total_steps_taken += 1
        self._update_epsilon()

        if training and random.random() < self.current_epsilon:
            return np.random.randint(self.num_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(stacked_frames_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def _update_epsilon(self):
        # Using exponential decay as in the original
        if self.eps_decay_steps <= 0: # Avoid division by zero or log of non-positive
            self.current_epsilon = self.eps_end
            return

        # Ensure eps_start is greater than eps_end to avoid issues with log
        if self.eps_start <= self.eps_end:
            self.current_epsilon = self.eps_end
            return

        decay_rate = np.exp(np.log(self.eps_end / self.eps_start) / self.eps_decay_steps)
        if self.total_steps_taken < self.eps_decay_steps :
             self.current_epsilon = self.eps_start * (decay_rate ** self.total_steps_taken)
        else:
             self.current_epsilon = self.eps_end

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size or self.total_steps_taken < self.learn_start_steps:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        if not transitions: # Safeguard if replay buffer sample returns empty
            return

        batch = list(zip(*transitions))

        state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(self.device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps_taken % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()
        print(f"Model loaded from {filepath}")