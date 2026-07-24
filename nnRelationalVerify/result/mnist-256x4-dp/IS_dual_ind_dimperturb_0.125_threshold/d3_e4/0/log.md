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
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040923, -0.0040685, -0.0040923, -0.0040685, -0.0000139, 0.0000139)
1: (-0.0059140, -0.0050200, -0.0059140, -0.0050200, -0.0005205, 0.0005205)
2: (0.9693665, 0.9704393, 0.9693665, 0.9704393, -0.0006247, 0.0006247)
3: (0.0203578, 0.0282708, 0.0203578, 0.0282708, -0.0046073, 0.0046073)
4: (-0.0028432, -0.0022414, -0.0028432, -0.0022414, -0.0003504, 0.0003504)
5: (0.0143968, 0.0150051, 0.0143968, 0.0150051, -0.0003542, 0.0003542)
6: (0.0046083, 0.0049042, 0.0046083, 0.0049042, -0.0001723, 0.0001723)
7: (-0.0151049, -0.0130541, -0.0151049, -0.0130541, -0.0011940, 0.0011940)
8: (0.0047457, 0.0063726, 0.0047457, 0.0063726, -0.0009473, 0.0009473)
9: (0.0062601, 0.0091864, 0.0062601, 0.0091864, -0.0017038, 0.0017038)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.75 + 1.29 = 3.04 seconds
status: Status.ADV_EXAMPLE
