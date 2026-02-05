import torch
from torch.distributions import Categorical
from network.PolicyNetwork import PolicyNetwork


class GetMultiplier:
    def __init__(self, multiplier_ls, policy, device):
        self.multiplier_ls = multiplier_ls
        self.device = device
        self.policy = policy
        # ###加载模型：
        # self.load_model(model_path)

    # def load_model(self, path):
    #     """加载模型和网络配置"""
    #     ###print(f"加载模型: {path}")
    #     model_info = torch.load(path)
    #     # 使用保存的配置重新创建策略网络
    #     self.policy = PolicyNetwork(
    #         model_info["state_dim"], model_info["action_dim"]
    #     ).to(self.device)
    #     # 加载模型参数
    #     self.policy.load_state_dict(model_info["state_dict"])
    #     self.policy.to(self.device)

    def select_multipler(self, state):
        """选择动作"""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        probs, state_value = self.policy(state)
        m = Categorical(probs)
        print(probs)
        action = m.sample().item()
        return self.multiplier_ls[action]

    def select_multipler_deterministic(self, state):
        """确定性地选择动作，推理的时候调用这个函数"""
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            probs, state_value = self.policy(state)  # 解包返回值，只使用概率
            print(probs)
            action = torch.argmax(probs).item()
        return self.multiplier_ls[action]
