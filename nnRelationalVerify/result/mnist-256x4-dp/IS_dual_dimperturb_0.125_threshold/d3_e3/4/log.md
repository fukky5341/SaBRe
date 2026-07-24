## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00027306


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9946499, 0.9959156, 0.9946499, 0.9959156, -0.0007870, 0.0007870)
1: (-0.0025970, -0.0022817, -0.0025970, -0.0022817, -0.0001961, 0.0001961)
2: (0.0020378, 0.0037090, 0.0020378, 0.0037090, -0.0010393, 0.0010393)
3: (-0.0029613, -0.0022006, -0.0029613, -0.0022006, -0.0004730, 0.0004730)
4: (0.0009223, 0.0012458, 0.0009223, 0.0012458, -0.0002011, 0.0002011)
5: (0.0015225, 0.0036244, 0.0015225, 0.0036244, -0.0013071, 0.0013071)
6: (0.0006209, 0.0011544, 0.0006209, 0.0011544, -0.0003318, 0.0003318)
7: (-0.0015311, -0.0001508, -0.0015311, -0.0001508, -0.0008584, 0.0008584)
8: (-0.0003693, 0.0003565, -0.0003693, 0.0003565, -0.0004514, 0.0004514)
9: (-0.0022773, -0.0014356, -0.0022773, -0.0014356, -0.0005234, 0.0005234)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.77 + 1.24 = 3.01 seconds
status: Status.ADV_EXAMPLE
