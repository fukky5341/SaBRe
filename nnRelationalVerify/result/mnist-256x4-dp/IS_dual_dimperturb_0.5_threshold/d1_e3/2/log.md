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
Threshold: 0.00109205


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0043747, -0.0041068, -0.0043747, -0.0041068, -0.0001969, 0.0001969)
1: (0.0024875, 0.0039709, 0.0024875, 0.0039709, -0.0010904, 0.0010904)
2: (0.0060948, 0.0094088, 0.0060948, 0.0094088, -0.0024360, 0.0024360)
3: (0.0033695, 0.0047660, 0.0033695, 0.0047660, -0.0010265, 0.0010265)
4: (1.0098225, 1.0152406, 1.0098225, 1.0152406, -0.0039826, 0.0039826)
5: (0.0041523, 0.0052063, 0.0041523, 0.0052063, -0.0007748, 0.0007748)
6: (-0.0125182, -0.0111465, -0.0125182, -0.0111465, -0.0010082, 0.0010082)
7: (-0.0104002, -0.0102252, -0.0104002, -0.0102252, -0.0001286, 0.0001286)
8: (-0.0031528, -0.0022051, -0.0031528, -0.0022051, -0.0006966, 0.0006966)
9: (-0.0071319, -0.0023875, -0.0071319, -0.0023875, -0.0034874, 0.0034874)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.31 + 1.42 = 2.73 seconds
status: Status.ADV_EXAMPLE
