## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00079287


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0015321, -0.0001000, -0.0015321, -0.0001000, -0.0011642, 0.0011642)
1: (-0.0043204, -0.0038198, -0.0043204, -0.0038198, -0.0004282, 0.0004282)
2: (0.0125672, 0.0145239, 0.0125672, 0.0145239, -0.0015441, 0.0015441)
3: (1.0079775, 1.0092640, 1.0079775, 1.0092640, -0.0012865, 0.0012865)
4: (-0.0039716, -0.0036444, -0.0039716, -0.0036444, -0.0002519, 0.0002519)
5: (0.0027729, 0.0038794, 0.0027729, 0.0038794, -0.0008955, 0.0008955)
6: (-0.0024804, -0.0023625, -0.0024804, -0.0023625, -0.0001178, 0.0001178)
7: (-0.0129841, -0.0112798, -0.0129841, -0.0112798, -0.0016752, 0.0016752)
8: (-0.0104414, -0.0068480, -0.0104414, -0.0068480, -0.0027356, 0.0027356)
9: (-0.0009315, 0.0008759, -0.0009315, 0.0008759, -0.0013741, 0.0013741)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 1.54 = 2.99 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -0.0007798, upper bound: 0.0007798
