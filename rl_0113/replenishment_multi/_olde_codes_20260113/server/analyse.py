from predictor import Predictor
model_path = "/home/work/apb-project/ais-deploy-demo-cache/replenishment-service-file/rl/id-20250624/id/20250624v0/model.onnx"
feature_map = "/home/work/apb-project/ais-deploy-demo-cache/replenishment-service-file/rl/id-20250624/id/20250624v0/model.json"
p = Predictor(model_path,None,feature_map,10)

# input_data_path = "/home/work/apb-project/ais-deploy-demo-cache/replenishment_wb/replenishment_ppo/req_test.text"
# with open(input_data_path, "r") as file:
#     json_data = file.read()

#     # 解析 JSON 数据
#     data = json.loads(json_data)

#     # 打印解析后的数据
#     print(data)