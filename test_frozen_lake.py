import gymnasium as gym


env = gym.make('FrozenLake-v1', render_mode="human", desc=None, map_name="4x4", is_slippery=True)
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
         observation, info = env.reset()

env.close()
