import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PolicyNetwork import PolicyNetwork
from atp_sim_sdk_py import replenish_env_multiple_actions, replenish_scene, rolling_env, strategy


class ReplenishAgentMultipleActions:
    def __init__(self, env, device, lr=3e-4, gamma=0.99, k_epochs=4, eps_clip=0.2):
        self.env = env
        self.device = device
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.num_products=env.num_products



        self.policy = PolicyNetwork(self.state_dim, self.num_products*self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip

        # 存储训练数据
        self.states = []
        self.actions = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.dones = []
        self.episode_rewards = []



    def select_action(self, state):
        """选择动作"""

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        probs, state_value = self.policy(state)

        # 将动作概率分布拆分为每个商品的分布
        action_probs = probs.view(self.num_products,self.action_dim)  # 重塑为 (num_products, action_dim)
        print(action_probs.shape)
        print(action_probs)

        # 为每个商品采样动作
        step_actions = []
        log_probs = []
        for probs in action_probs:
            m = Categorical(probs)  # 创建分类分布
            action = m.sample()  # 采样动作
            log_prob = m.log_prob(action)  # 计算对数概率
            step_actions.append(action.item())
            log_probs.append(log_prob)
        print(step_actions)
        ###更新记忆数据
        self.states.append(state.detach())
        self.actions.append(step_actions)
        self.logprobs.append(torch.stack(log_probs))
        self.state_values.append(state_value.detach())

        return step_actions




    def update(self):
        """更新策略网络"""
        print(self.states)
        print(self.actions)
        states = torch.stack(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        logprobs = torch.stack(self.logprobs).to(self.device)
        state_values = torch.cat(self.state_values).squeeze().to(self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(self.device)
        print("################################################")
        print(actions)
        print(actions.shape)
        print(states)

        # 计算回报和优势值
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - state_values.detach()
        # 标准化优势值
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        print(len(states))

        # PPO更新
        for _ in range(self.k_epochs):
            probs, state_values_new = self.policy(states)# 获取动作的概率分布
            #########拆分为每个商品对应的概率
            probs = probs.view(len(states), self.num_products, self.action_dim)
            m = Categorical(probs)
            new_logprobs = m.log_prob(actions)
            entropy = m.entropy().mean()

            # cal ratio，对所有品的概率值取均值
            new_logprobs=torch.mean(new_logprobs, dim=1)
            logprobs=torch.mean(logprobs,dim=1)

            ratios = torch.exp(new_logprobs - logprobs)
            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2).mean()
                + 0.5 * nn.MSELoss()(state_values_new.squeeze(), returns)
                - 0.01 * entropy
            )

            print("####################probs#####################")
            print(_)
            print("probs: ", probs)
            print("probs: ", probs.shape)
            print("new_logprobs: ", new_logprobs.shape)
            print("logprobs ",logprobs)
            print("m: ",m)
            print(m.entropy().shape)
            #print(m.entropy())
            print("entropy: ", entropy.shape)
            print("ratios: ", ratios.shape)
            print("advantages: ", advantages.shape)
            print("loss ",loss)

            ###take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 清空缓存
        self.states = []
        self.actions = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.dones = []
    def update_by_batch(self,batch_size=64):
        """更新策略网络"""
        print(self.states)
        print(self.actions)
        states = torch.stack(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        logprobs = torch.stack(self.logprobs).to(self.device)
        state_values = torch.cat(self.state_values).squeeze().to(self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(self.device)
        print("################################################")
        print(actions)
        print(actions.shape)
        print(states)

        # 计算回报和优势值
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - state_values.detach()
        # 标准化优势值
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 将所有经验数据合并为一个数据集
        dataset = torch.utils.data.TensorDataset(states, actions, logprobs, advantages, returns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # PPO更新
        for _ in range(self.k_epochs):
            for batch in dataloader:
                batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns = batch
                # 计算新策略的对数概率和状态值
                probs, state_values_new = self.policy(batch_states)# 获取动作的概率分布
                #########拆分为每个商品对应的概率
                probs = probs.view(len(batch_states), self.num_products, self.action_dim)
                m = Categorical(probs)
                batch_new_logprobs = m.log_prob(batch_actions)
                entropy = m.entropy().mean()

                # cal ratio，对所有品的概率值取均值
                batch_new_logprobs=torch.mean(batch_new_logprobs, dim=1)
                batch_logprobs=torch.mean(batch_logprobs,dim=1)

                ratios = torch.exp(batch_new_logprobs - batch_logprobs)
                # Finding Surrogate Loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                # final loss of clipped objective PPO
                loss = (
                -torch.min(surr1, surr2).mean()
                + 0.5 * nn.MSELoss()(state_values_new.squeeze(), returns)
                - 0.01 * entropy)


                ###更新网络
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # 清空缓存
        self.states = []
        self.actions = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.dones = []


    def plot_rewards(self):
        """绘制奖励趋势图"""
        data = np.array(self.episode_rewards)
        plt.figure(figsize=(12, 6))
        plt.plot(data, linewidth=1)
        plt.title("奖励趋势")
        plt.xlabel("回合数")
        plt.ylabel("奖励值")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.show()

    def train(self, max_episodes=3000):
        """训练智能体"""
        for episode in range(max_episodes):

            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(state)
                step_result = self.env.step(action)

                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result

                self.rewards.append(reward)
                self.dones.append(done)
                state = next_state
                total_reward += reward

            self.update()
            self.episode_rewards.append(total_reward)

            if (episode + 1) % 10 == 0:
                print(f"回合 {episode + 1}\t总奖励: {total_reward}")
            print(f"回合 {episode + 1}\t总奖励: {total_reward}")

    def evaluate(self, state):
        """评估状态"""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs, state_value = self.policy(state)
        return probs, state_value

    def save_model(self, path):
        """保存模型"""
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        """加载模型"""
        self.policy.load_state_dict(torch.load(path))
        self.policy.to(self.device)


if __name__ == "__main__":
    df = pd.read_csv("./data/test_data.csv")
    st = strategy.StrategyB()
    scene = replenish_scene.CacheOrder(st)
    roller = rolling_env.RollingEnv(scene, rts_day=14)
    env = replenish_env_multiple_actions.ReplenishEnv(df, roller)
    agent = ReplenishAgentMultipleActions(env, device="cpu")
    # state = env.reset()
    # print(agent)
    # print(agent.select_action(state))
    # actions=[1,2]
    # print(actions)
    # print(env.step(actions))

    agent.train(5)
    agent.save_model("rep_model.pth")
    agent.plot_rewards()



