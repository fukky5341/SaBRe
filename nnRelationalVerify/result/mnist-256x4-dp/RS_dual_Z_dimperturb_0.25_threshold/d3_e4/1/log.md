## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00167283


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0013676, -0.0001661, -0.0013676, -0.0001661, -0.0009322, 0.0009322)
1: (-0.0077811, -0.0047321, -0.0077811, -0.0047321, -0.0023655, 0.0023655)
2: (0.0302026, 0.0320942, 0.0302026, 0.0320942, -0.0014675, 0.0014675)
3: (-0.0003655, 0.0031666, -0.0003655, 0.0031666, -0.0027403, 0.0027403)
4: (-0.0068077, -0.0037063, -0.0068077, -0.0037063, -0.0024061, 0.0024061)
5: (0.0111596, 0.0123343, 0.0111596, 0.0123343, -0.0009114, 0.0009114)
6: (-0.0000758, 0.0044070, -0.0000758, 0.0044070, -0.0034778, 0.0034778)
7: (0.9780063, 0.9811431, 0.9780063, 0.9811431, -0.0024336, 0.0024336)
8: (-0.0101450, -0.0067818, -0.0101450, -0.0067818, -0.0026092, 0.0026092)
9: (-0.0005198, 0.0017018, -0.0005198, 0.0017018, -0.0017235, 0.0017235)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.87 + 1.55 = 3.42 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0015768, upper bound: 0.0015768
