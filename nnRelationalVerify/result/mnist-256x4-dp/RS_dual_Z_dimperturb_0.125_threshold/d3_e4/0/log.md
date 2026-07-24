## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00020622


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040953, -0.0040694, -0.0040953, -0.0040694, -0.0000149, 0.0000149)
1: (-0.0060234, -0.0050550, -0.0060234, -0.0050550, -0.0005574, 0.0005574)
2: (0.9692352, 0.9703972, 0.9692352, 0.9703972, -0.0006688, 0.0006688)
3: (0.0193896, 0.0279607, 0.0193896, 0.0279607, -0.0049333, 0.0049333)
4: (-0.0028196, -0.0021677, -0.0028196, -0.0021677, -0.0003752, 0.0003752)
5: (0.0144206, 0.0150795, 0.0144206, 0.0150795, -0.0003792, 0.0003792)
6: (0.0045721, 0.0048926, 0.0045721, 0.0048926, -0.0001844, 0.0001844)
7: (-0.0150245, -0.0128032, -0.0150245, -0.0128032, -0.0012785, 0.0012785)
8: (0.0048094, 0.0065717, 0.0048094, 0.0065717, -0.0010143, 0.0010143)
9: (0.0063748, 0.0095444, 0.0063748, 0.0095444, -0.0018243, 0.0018243)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.76 + 1.27 = 3.03 seconds
status: Status.ADV_EXAMPLE
