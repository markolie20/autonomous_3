
# Class `DQN_CNN`
Deep-Q-Network **Convolutional Neural Network** (CNN) that transforms a stack of raw game frames into a vector of **Q-values**—one scalar per discrete action.  
Because Atari frames are images, convolutions extract spatial features, and the fully-connected layers fuse them into a single estimate of “how good” each action is.

---

## Method `__init__(self, input_channels, num_actions, h, w)`
Creates all layers and *computes* the exact flatten size so you never hard-code it.

**Parameters**

| name | type | explanation (with jargon decoded) |
|------|------|-----------------------------------|
| `input_channels` | `int` | Number of stacked frames that arrive as channels (usually 4). |
| `num_actions`    | `int` | Size of the game’s **action space**; the network will output that many Q-values. |
| `h`, `w`         | `int` | Height and width of a single pre-processed frame (e.g. 84 px). |

**Math and code**

| code line | jargon & purpose |
|-----------|------------------|
| ```self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)``` | **Convolution** with 32 learnable **filters** (aka kernels) of size 8×8. A stride of 4 means the filter jumps 4 pixels at a time ⇒ heavy down-sampling. |
| ```self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)``` | Second conv: doubles the channels to 64, kernel 4×4, stride 2 ⇒ extracts finer patterns while shrinking the map. |
| ```self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)``` | Third conv: keeps 64 channels; stride 1 preserves spatial resolution for one final pass. |
| ```def conv2d_size_out(size, kernel_size, stride): return (size - (kernel_size - 1) - 1) // stride + 1``` | Exact PyTorch output-shape formula: ⌊(in − kernel)/stride⌋ + 1 (no padding). |
| ```convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)``` | Cascades the formula 3 times to get final **width** after conv3. |
| ```convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)``` | Same for **height**. |
| ```linear_input_size = convw * convh * 64``` | Volume = width × height × channels (64). |
| ```self.fc1 = nn.Linear(linear_input_size, 512)``` | First **fully-connected** layer (“dense” layer) – mixes all spatial features. |
| ```self.fc2 = nn.Linear(512, num_actions)``` | Output layer: one Q-value for every action. |

---

## Method `forward(self, x)`
Feeds a batch of states through the layers and **returns** `Q(s,·)`.

**Parameters**

| name | type | explanation |
|------|------|-------------|
| `x` | `torch.Tensor` (shape `[batch, C, H, W]`) | The mini-batch of stacked, normalised frames. |

**Math and code**

```python
x = F.relu(self.conv1(x))          # ReLU = max(0, z) adds non-linearity
x = F.relu(self.conv2(x))
x = F.relu(self.conv3(x))
x = x.view(x.size(0), -1)          # flatten all but batch dim
x = F.relu(self.fc1(x))
return self.fc2(x)                 # shape: [batch, num_actions]
```

**Jargon notes**

* **ReLU** (Rectified Linear Unit) outputs `0` if the input is negative, otherwise the input itself.  
* `.view(..., -1)` reshapes a tensor without copying memory; `-1` means “infer this dimension”.

---

# Class `ReplayBuffer`
Implements **experience replay**—a fixed-size, first-in-first-out (FIFO) memory that stores past transitions so learning can sample them **i.i.d.** (independently and identically distributed). This breaks the strong correlation between consecutive frames that otherwise destabilises Q-learning.

---

## Method `__init__(self, capacity)`
Creates a `collections.deque` that will automatically discard oldest entries when full.

_No math._

---

## Method `add(self, state, action, reward, next_state, done)`
Pushes one transition.

```python
self.buffer.append((state, action, reward, next_state, done))
```

* `state`, `next_state`: stacked frames (shape `[C,H,W]`).  
* `action`: `int`.  
* `reward`: `float`.  
* `done`: `bool` (converted to `1.0` / `0.0` later).

---

## Method `sample(self, batch_size)`
Uniformly picks `batch_size` distinct elements:

```python
return random.sample(self.buffer, batch_size)
```

Uniform sampling means every stored transition is equally likely, an assumption made by stochastic gradient descent.

---

## Method `__len__(self)`
Python “dunder” so you can call `len(replay_buffer)`.

```python
return len(self.buffer)
```

---

# Class `DQNAgent`
High-level wrapper that glues the CNN, target network, replay buffer, ε-greedy policy and training loop into a single object.

---

## Method `__init__(...)`
Sets hyper-parameters, chooses **device** (CPU vs. CUDA GPU), builds the two networks and the optimiser, and initialises variables that track exploration and frame stacking.

| code fragment | explanation of jargon |
|---------------|-----------------------|
| ```self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")``` | **Device** is the hardware running the tensor ops; CUDA uses NVIDIA GPUs for large speedups. |
| ```self.policy_net = DQN_CNN(...)``` | The network we *train*; often called θ. |
| ```self.target_net = DQN_CNN(...)``` | Frozen copy (θ′) used to compute stable targets. |
| ```self.target_net.eval()``` | Switches off dropout/batch-norm (not present here but good practice). |
| ```self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)``` | **Adam** is an adaptive learning-rate variant of SGD. |
| ```self.current_epsilon = eps_start``` | ε controls exploration: higher ⇒ more random moves. |
| ```self.total_steps_taken = 0``` | Counts how many *agent* actions have been decided; drives ε-decay and target sync. |

---

## Method `_preprocess_frame(self, frame)`
Exactly recreates the preprocessing from the original Atari DQN:

1. **Grayscale** conversion via `cv2.cvtColor`.  
2. **Resize** to 84×84 with area interpolation (better down-sampling).  
3. Returns an 8-bit array; pixel intensities stay 0-255.

---

## Method `get_stacked_frames(self, raw_observation)`
Keeps the last `frame_stack_size` frames (4) to introduce temporal information (e.g., ball velocity). **Stacking** is done on the channel dimension so the CNN treats them as coloured layers.

Key steps:

```python
processed_frame = self._preprocess_frame(raw_observation)
if self.new_episode_started:
    self._frame_stack.clear()           # first observation of a new life
    for _ in range(self.frame_stack_size):
        self._frame_stack.append(processed_frame)  # duplicate so size == 4
    self.new_episode_started = False
else:
    self._frame_stack.append(processed_frame)
stacked_np_array = np.array(list(self._frame_stack), dtype=np.float32) / 255.0
return stacked_np_array
```

* Division by 255.0 normalises pixels to `[0.0, 1.0]`, which speeds training.

---

## Method `reset_episode_state(self)`
Called when the environment signals **terminal**. Simply sets a flag so the next `get_stacked_frames` knows to flush the buffer.

---

## Method `act(self, stacked_frames_state, training=True)`
Picks an action according to ε-greedy policy.

| code line | vocabulary |
|-----------|------------|
| ```self.total_steps_taken += 1``` | Advances the global step counter. |
| ```self._update_epsilon()``` | Decays exploration. |
| ```if training and random.random() < self.current_epsilon:``` | With prob ε choose a **random** integer action. |
| ```state_tensor = torch.tensor(stacked_frames_state, dtype=torch.float32).unsqueeze(0)``` | `.unsqueeze(0)` adds a batch dim (`1, C, H, W`). |
| ```q_values = self.policy_net(state_tensor)``` | Forward pass. |
| ```return q_values.argmax(dim=1).item()``` | `argmax` returns index of the highest Q-value ⇒ **greedy** action. |

---

## Method `_update_epsilon(self)`
Uses **exponential** decay so early on ε ≈ 1 (pure exploration), gradually shrinking to `eps_end`.

```python
decay_rate = np.exp(np.log(self.eps_end / self.eps_start) / self.eps_decay_steps)
if self.total_steps_taken < self.eps_decay_steps:
    self.current_epsilon = self.eps_start * (decay_rate ** self.total_steps_taken)
else:
    self.current_epsilon = self.eps_end
```

Mathematically:  
ε(t) = ε₀ · (ε_end / ε₀)^(t / N), N = `eps_decay_steps`.

---

## Method `train_step(self)`
Runs **one** gradient update provided that

* Enough transitions are in the buffer (`len ≥ batch_size`), **and**  
* We passed the warm-up period (`total_steps_taken ≥ learn_start_steps`).

Full flow, line by line:

1. **Sample & transpose**

   ```python
   transitions = self.replay_buffer.sample(self.batch_size)
   batch = list(zip(*transitions))   # rearrange so each entry is a list of same-type items
   ```

2. **Create tensors on GPU/CPU**

   ```python
   state_batch      = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(self.device)
   action_batch     = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1).to(self.device)
   reward_batch     = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(self.device)
   next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(self.device)
   done_batch       = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(self.device)
   ```
   *`unsqueeze(1)` turns shape `[B]` into `[B,1]` to align with gather operations.*

3. **Predict Q(s,a) with the *online* network**

   ```python
   current_q_values = self.policy_net(state_batch).gather(1, action_batch)
   ```

4. **Predict maxₐ′Q(s′,a′) with the *target* network (no grad)**

   ```python
   with torch.no_grad():
       next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
   ```

5. **Compute Bellman target**

   ```python
   expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
   ```
   *Because `done` is 1 when the episode ends, `(1 - done)` nullifies the bootstrap term at terminals.*

6. **Loss: Huber (a.k.a. smooth L1)**

   ```python
   loss = F.smooth_l1_loss(current_q_values, expected_q_values)
   ```
   **Huber formula in code notation**:  
   ```
   diff = |current_q - expected_q|
   loss = 0.5 * diff**2              if diff < 1
          1.0 * (diff - 0.5)         otherwise
   ```
   It is quadratic near 0 (like MSE) but linear for large errors, making it less sensitive to outliers.

7. **Optimisation step**

   ```python
   self.optimizer.zero_grad()
   loss.backward()
   # optional: torch.nn.utils.clip_grad_value_(..., 100)
   self.optimizer.step()
   ```

8. **Target-network sync**

   ```python
   if self.total_steps_taken % self.target_update_frequency == 0:
       self.target_net.load_state_dict(self.policy_net.state_dict())
   ```
   This is a *hard* update (copy); alternatives are “soft” Polyak averaging.

---

## Method `save_model(self, filepath)`
Writes **parameters only** (no optimizer state) with:

```python
torch.save(self.policy_net.state_dict(), filepath)
```

---

## Method `load_model(self, filepath)`
Loads weights, mirrors them to the target network, and switches both to inference mode:

```python
self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
self.target_net.load_state_dict(self.policy_net.state_dict())
self.policy_net.eval()
self.target_net.eval()
```

---

### Recap of Jargon Explained
* **Q-value Q(s,a)** – expected discounted return when taking action a in state s and following the policy thereafter.  
* **Target network** – frozen copy θ′ that stabilises learning by providing fixed Q-value targets for a while.  
* **Replay buffer** – stores past experiences so updates are based on a random mixture of many episodes.  
* **ε-greedy** – exploration strategy: with probability ε pick a uniform random action, otherwise exploit the network’s argmax.  
* **Stride** – step size of the convolution filter; bigger stride ⇒ lower resolution but fewer FLOPs.  
* **Kernel / filter** – the small weight matrix slid over the image to detect patterns.  
* **Huber loss** – combines MSE (for small errors) and MAE (for large ones) to reduce gradient explosions.  
* **Unsqueeze** – adds a dimension of length 1, often used to make tensors broadcast-compatible.  
* **Flatten** – collapse spatial dimensions into one long vector before dense layers.  
* **Adam** – optimiser that adapts the learning rate per parameter using running averages of gradients and their squares.

With these components the agent faithfully implements the seminal 2015 DeepMind **DQN** algorithm: stable targets, experience replay, frame stacking, an ε-greedy behaviour policy, and a CNN approximator.
