import numpy as np

class BaselineAgent():
  def __init__(self, player_name):
    self.name = player_name
  
  def act(self, observation):
    return np.random.randint(6)
