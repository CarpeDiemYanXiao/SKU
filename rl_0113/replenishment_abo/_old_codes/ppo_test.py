import gym
import torch
from PPOAgent import PPOAgent
from gym.wrappers import RecordVideo
import argparse


def train_with_no_ui():
    env = gym.make('CartPole-v1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    agent = PPOAgent(env, device)
    agent.train(3000)
    agent.save_model("ppo_model.pth")
    env.close()
    agent.plot_rewards()

def train_with_ui():
    # 创建环境并启用视频录制
    env = gym.make('CartPole-v1', render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    agent = PPOAgent(env, device)
    agent.train(500)
    agent.save_model("ppo_model.pth")
    # 训练后进行展示
    print("\n开始展示训练结果...")
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        env.render()  # 渲染当前帧
        action = agent.select_action(state)
        step_result = env.step(action)
        
        if len(step_result) == 5:
            state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            state, reward, done, _ = step_result
            
        total_reward += reward
    
    print(f"展示完成，总奖励: {total_reward}")
    env.close()
    agent.plot_rewards()


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='PPO训练程序')
    parser.add_argument('--ui', action='store_true', 
                       help='使用界面模式训练，默认为无界面模式')
    
    args = parser.parse_args()
    
    if args.ui:
        train_with_ui()
    else:
        train_with_no_ui()  



