import json
from locust import FastHttpUser, task

"""
压测主文件：定义入参格式
"""

# TODO: EDT Test - BATCH PREDICTION
json_path = "/home/work/apb-project/rl_req.json"
with open(json_path,'rb') as f:
    inputs = json.load(f)

raw = inputs["data"]["input"][0]

cnt = 16
inputs["data"]["input"] = [raw for i in range(cnt)]
req_data = json.dumps(inputs)

# inputs = {"input": [eval(f"raw_input") for i in range(cnt)], "ModelVersion": "20241029"}


# def build_request():
#     req = json.dumps({"logid": 1234567, "clientip": "", "data": inputs})
#     return req


class MyUser(FastHttpUser):
    @task
    def process(self):
        #req_data = build_request()
        with self.client.post("/api/process", data=req_data) as res:
            if res.status_code != 200:
                print("Didn't detect bad response, got: " + str(res.status_code))
