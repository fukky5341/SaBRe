## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000149


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.0035672, 1.0040617, 1.0035672, 1.0040617, -0.0001255, 0.0001255)
1: (-0.0003751, -0.0002519, -0.0003751, -0.0002519, -0.0000313, 0.0000313)
2: (-0.0087192, -0.0080663, -0.0087192, -0.0080663, -0.0001657, 0.0001657)
3: (0.0023983, 0.0026955, 0.0023983, 0.0026955, -0.0000754, 0.0000754)
4: (-0.0011597, -0.0010333, -0.0011597, -0.0010333, -0.0000321, 0.0000321)
5: (-0.0120070, -0.0111858, -0.0120070, -0.0111858, -0.0002084, 0.0002084)
6: (0.0043799, 0.0045883, 0.0043799, 0.0045883, -0.0000529, 0.0000529)
7: (0.0081945, 0.0087338, 0.0081945, 0.0087338, -0.0001369, 0.0001369)
8: (0.0047453, 0.0050289, 0.0047453, 0.0050289, -0.0000720, 0.0000720)
9: (-0.0076951, -0.0073662, -0.0076951, -0.0073662, -0.0000835, 0.0000835)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 1.23 = 2.59 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0001255, upper bound: 0.0001255
