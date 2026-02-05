from PolicyNetwork import PolicyNetwork
policy_model = PolicyNetwork(state_dim = 8, action_dim = 16, hidden_dim = 64)

from normalization import Normalization
import torch
import onnx
from onnx2torch import convert
path = "/home/work/apb-project/ais-deploy-demo-cache/replenishment_wb/replenishment_ppo/output/cache_rp/wb_v1/20250624v0/model.onnx"
# model_info = torch.load(path, "cpu", weights_only=True) 
model_info = onnx.load(path)
model_info = convert(model_info)
policy_model.load_state_dict(model_info["state_dict"])
policy_model.to("cpu")
# guiyihua
state_norm = Normalization(shape=model_info["state_dim"], config=None)
state_norm.running_ms.mean = np.array(model_info["state_norm_mean"])
state_norm.running_ms.std = np.array(model_info["state_norm_std"])
