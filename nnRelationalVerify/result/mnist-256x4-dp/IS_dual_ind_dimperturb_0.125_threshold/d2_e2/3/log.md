## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0018212


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0002204, -0.0000605, -0.0002204, -0.0000605, -0.0001117, 0.0001117)
1: (0.0002883, 0.0010367, 0.0002883, 0.0010367, -0.0005230, 0.0005230)
2: (0.0147874, 0.0159082, 0.0147874, 0.0159082, -0.0007832, 0.0007832)
3: (0.0004926, 0.0013354, 0.0004926, 0.0013354, -0.0005890, 0.0005890)
4: (-0.0039253, -0.0031478, -0.0039253, -0.0031478, -0.0005433, 0.0005433)
5: (0.0084299, 0.0092712, 0.0084299, 0.0092712, -0.0005879, 0.0005879)
6: (0.0094342, 0.0097517, 0.0094342, 0.0097517, -0.0002219, 0.0002219)
7: (-0.0185261, -0.0166998, -0.0185261, -0.0166998, -0.0012762, 0.0012762)
8: (0.9707114, 0.9759440, 0.9707114, 0.9759440, -0.0036566, 0.0036566)
9: (0.0047232, 0.0062611, 0.0047232, 0.0062611, -0.0010747, 0.0010747)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.31 + 1.18 = 2.49 seconds
status: Status.ADV_EXAMPLE
