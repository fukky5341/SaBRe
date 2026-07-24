## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 8.752e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041968, -0.0041919, -0.0041968, -0.0041919, -0.0000049, 0.0000049)
1: (-0.0098254, -0.0096409, -0.0098254, -0.0096409, -0.0001845, 0.0001845)
2: (0.9646726, 0.9648941, 0.9646726, 0.9648941, -0.0002215, 0.0002215)
3: (-0.0142634, -0.0126299, -0.0142634, -0.0126299, -0.0011935, 0.0011935)
4: (0.0002675, 0.0003918, 0.0002675, 0.0003918, -0.0001242, 0.0001242)
5: (0.0175407, 0.0176721, 0.0175407, 0.0176721, -0.0001313, 0.0001313)
6: (0.0032986, 0.0033750, 0.0032986, 0.0033750, -0.0000764, 0.0000764)
7: (-0.0045051, -0.0040727, -0.0045051, -0.0040727, -0.0004324, 0.0004324)
8: (0.0131550, 0.0134909, 0.0131550, 0.0134909, -0.0003358, 0.0003358)
9: (0.0213852, 0.0219892, 0.0213852, 0.0219892, -0.0005558, 0.0005558)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.23 + 1.14 = 2.37 seconds
status: Status.ADV_EXAMPLE
