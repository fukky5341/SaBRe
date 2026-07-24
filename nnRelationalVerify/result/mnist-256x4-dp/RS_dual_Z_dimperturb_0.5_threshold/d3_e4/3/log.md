## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.603988435


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.4074745, 1.0653124, 0.4074745, 1.0653124, -0.6578379, 0.6578379)
1: (-0.2808680, 0.2742082, -0.2808680, 0.2742082, -0.5550762, 0.5550762)
2: (-0.1965990, 0.3700640, -0.1965990, 0.3700640, -0.5666630, 0.5666630)
3: (-0.2045170, 0.2769909, -0.2045170, 0.2769909, -0.4815079, 0.4815079)
4: (-0.2941673, 0.2552224, -0.2941673, 0.2552224, -0.5493897, 0.5493897)
5: (-0.3158750, 0.4136090, -0.3158750, 0.4136090, -0.7294840, 0.7294840)
6: (-0.2227369, 0.3051631, -0.2227369, 0.3051631, -0.5279000, 0.5279000)
7: (-0.3064720, 0.3126367, -0.3064720, 0.3126367, -0.6191087, 0.6191087)
8: (-0.2931817, 0.3803814, -0.2931817, 0.3803814, -0.6735631, 0.6735631)
9: (-0.2854504, 0.3676052, -0.2854504, 0.3676052, -0.6530557, 0.6530557)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.74 + 2.35 = 4.09 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.5419274, upper bound: 0.5419274
