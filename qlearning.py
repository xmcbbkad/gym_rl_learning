import numpy as np
import gymnasium as gym
from collections import defaultdict

"""配置参数"""
class Config:
    def __init__(self):
        self.env_name = 'FrozenLake-v1'  # 环境名称
        self.train_eps = 50000  # 训练回合数
        self.test_eps = 20  # 测试回合数
        self.max_steps = 50  # 每个回合最大步数
        self.epsilon_start = 0.75  # e-greedy策略中epsilon的初始值
        self.epsilon_end = 0.1  # e-greedy策略中epsilon的最终值
        self.epsilon_decay = 300  # e-greedy策略中epsilon的衰减率
        self.epsilon_denominator = 1000  # e-greedy策略中epsilon的衰减率
        self.gamma = 0.9  # 折扣因子
        self.lr = 0.2  # 学习率
        self.seed = 114514  # 随机种子

    def decay_func(self, sample_count):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * max(np.exp(-sample_count / self.epsilon_decay), (1 - sample_count / self.epsilon_denominator))


"""Q-learning"""
class QLearning(object):
    def __init__(self, n_states, n_actions, cfg):
        self.n_states = n_states
        self.n_actions = n_actions
        self.cfg = cfg
        self.sample_count = 0
        self.Q_table = defaultdict(lambda: np.zeros(n_actions))  # 嵌套字典来储存状态->动作
        self.epsilon = 0.

    """action采样"""
    def sample_action(self, state):
        self.sample_count += 1
        epsilon = self.cfg.decay_func(self.sample_count)
        #epsilon = 0.1
        self.epsilon = epsilon
        sample_idx = np.random.uniform(0, 1)
        if sample_idx > epsilon:

            max_indices = np.where(self.Q_table[str(state)] == np.max(self.Q_table[str(state)]))[0]
            action = np.random.choice(max_indices)

            #action = np.argmax(self.Q_table[str(state)])
        else:
            action = np.random.choice(self.n_actions)
        return action

    """预测动作"""
    def predict_action(self, state):
        action = np.argmax(self.Q_table[str(state)])
        return action

    """更新Q表"""
    def update_Q(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[str(state)][action]
        if done:
            Q_target = reward
        else:
            Q_target = reward + self.cfg.gamma * np.max(self.Q_table[str(next_state)])
        self.Q_table[str(state)][action] += self.cfg.lr * (Q_target - Q_predict)

"""定义训练和测试函数"""
def train(cfg, agent, env):
    print("开始训练")
    for i_ep in range(cfg.train_eps):
        ep_reward = 0.
        state = env.reset(seed = cfg.seed)
        if isinstance(state, tuple):
            state = state[0]
        steps = cfg.max_steps
        for i_step in range(cfg.max_steps):
            action = agent.sample_action(state)
            observation, reward, done, truncated, info = env.step(action)
            if done and reward == 0:
                reward = -10
            #env.render()
            

            #print(state)
            #print(type(state))
            agent.update_Q(state, action, reward, observation, done)
            state = observation
            ep_reward += reward
            if done:
                steps = i_step + 1
                break
        print(f"episode:{i_ep + 1}, reward:{ep_reward}, steps:{steps}, epsilon:{agent.epsilon}")
        print(agent.Q_table)
    print("训练完成")


def test(cfg, agent, env):
    print("开始测试")
    rewards = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0.
        state = env.reset(seed=cfg.seed)
        for i_step in range(cfg.max_steps):
            action = agent.predict_action(state)
            observation, reward, done, truncated, info = env.step(action)
            env.render()
            agent.update_Q(state, action, reward, observation, done)
            state = observation
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
    print(f"测试完成, average_reward = {np.mean(rewards)}")

"""创建环境和智能体"""
cfg = Config()
#env = gym.make(cfg.env_name, is_slippery = True,  render_mode="human")
env = gym.make(cfg.env_name, is_slippery = True)
agent = QLearning(env.observation_space.n, env.action_space.n, cfg)

train(cfg, agent, env)  # 训练
test(cfg, agent, env)  # 测试
