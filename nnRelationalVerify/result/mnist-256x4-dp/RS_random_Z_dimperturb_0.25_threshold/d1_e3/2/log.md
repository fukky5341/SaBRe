## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00063364


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0043345, -0.0042216, -0.0043345, -0.0042216, -0.0000719, 0.0000719)
1: (0.0031230, 0.0037483, 0.0031230, 0.0037483, -0.0003983, 0.0003983)
2: (0.0065921, 0.0079890, 0.0065921, 0.0079890, -0.0008899, 0.0008899)
3: (0.0039678, 0.0045564, 0.0039678, 0.0045564, -0.0003750, 0.0003750)
4: (1.0121436, 1.0144274, 1.0121436, 1.0144274, -0.0014549, 0.0014549)
5: (0.0046038, 0.0050481, 0.0046038, 0.0050481, -0.0002830, 0.0002830)
6: (-0.0123123, -0.0117342, -0.0123123, -0.0117342, -0.0003683, 0.0003683)
7: (-0.0103739, -0.0103002, -0.0103739, -0.0103002, -0.0000470, 0.0000470)
8: (-0.0027467, -0.0023473, -0.0027467, -0.0023473, -0.0002545, 0.0002545)
9: (-0.0064199, -0.0044201, -0.0064199, -0.0044201, -0.0012740, 0.0012740)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 1.24 = 2.60 seconds
status: Status.ADV_EXAMPLE
