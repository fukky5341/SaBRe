## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0004916


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0001701, 0.0007485, 0.0001701, 0.0007485, -0.0004026, 0.0004026)
1: (0.9942942, 0.9955190, 0.9942942, 0.9955190, -0.0008526, 0.0008526)
2: (-0.0079208, -0.0076315, -0.0079208, -0.0076315, -0.0002014, 0.0002014)
3: (0.0029009, 0.0036245, 0.0029009, 0.0036245, -0.0005037, 0.0005037)
4: (0.0027790, 0.0037222, 0.0027790, 0.0037222, -0.0006565, 0.0006565)
5: (0.0037654, 0.0051360, 0.0037654, 0.0051360, -0.0009540, 0.0009540)
6: (-0.0009222, 0.0003452, -0.0009222, 0.0003452, -0.0008822, 0.0008822)
7: (-0.0074990, -0.0069127, -0.0074990, -0.0069127, -0.0004081, 0.0004081)
8: (0.0080907, 0.0081679, 0.0080907, 0.0081679, -0.0000537, 0.0000537)
9: (-0.0031438, -0.0023066, -0.0031438, -0.0023066, -0.0005827, 0.0005827)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 1.19 = 2.52 seconds
status: Status.ADV_EXAMPLE
