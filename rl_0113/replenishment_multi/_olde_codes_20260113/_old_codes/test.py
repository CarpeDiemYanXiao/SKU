# # import torch
# #
# # # 定义类别的概率分布
# # probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4])
# #
# # # 创建 Categorical 分布
# # categorical_dist = torch.distributions.Categorical(probs=probabilities)
# import torch
# from torch.distributions import Categorical
#
# probs = torch.tensor([0.1, 0.2, 0.3, 0.4])  # 每个动作的概率
# m = Categorical(probs)
#
# print(m.probs)
# dic = {}
# sample = 0
#
# for i in range(1000):
#     sample = m.sample()
#     print(sample.item(), sample.detach(), m.log_prob(sample).item())
#     if dic.get(sample.item()) is not None:
#         dic[sample.item()] += 1
#     else:
#         dic[sample.item()] = 0
# print(dic)


