## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00013584


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0038915, -0.0025363, -0.0038915, -0.0025363, -0.0013552, 0.0013552)
1: (0.0055330, 0.0065003, 0.0055330, 0.0065003, -0.0009673, 0.0009673)
2: (0.0108027, 0.0133478, 0.0108027, 0.0133478, -0.0020900, 0.0020900)
3: (-0.0040603, -0.0029040, -0.0040603, -0.0029040, -0.0011339, 0.0011339)
4: (0.0047767, 0.0051615, 0.0047767, 0.0051615, -0.0002137, 0.0002137)
5: (-0.0018420, -0.0010054, -0.0018420, -0.0010054, -0.0008367, 0.0008367)
6: (-0.0057812, -0.0053683, -0.0057812, -0.0053683, -0.0004129, 0.0004129)
7: (-0.0031400, -0.0023510, -0.0031400, -0.0023510, -0.0007890, 0.0007890)
8: (-0.0033049, -0.0016223, -0.0033049, -0.0016223, -0.0016826, 0.0016826)
9: (1.0004455, 1.0006939, 1.0004455, 1.0006939, -0.0002484, 0.0002484)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.28 + 1.23 = 2.51 seconds
status: Status.ADV_EXAMPLE
