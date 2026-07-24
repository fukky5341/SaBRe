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
0: (1.0036199, 1.0040231, 1.0036199, 1.0040231, -0.0001178, 0.0001178)
1: (-0.0003620, -0.0002615, -0.0003620, -0.0002615, -0.0000294, 0.0000294)
2: (-0.0086681, -0.0081358, -0.0086681, -0.0081358, -0.0001556, 0.0001556)
3: (0.0024299, 0.0026722, 0.0024299, 0.0026722, -0.0000708, 0.0000708)
4: (-0.0011498, -0.0010468, -0.0011498, -0.0010468, -0.0000301, 0.0000301)
5: (-0.0119428, -0.0112732, -0.0119428, -0.0112732, -0.0001957, 0.0001957)
6: (0.0044021, 0.0045720, 0.0044021, 0.0045720, -0.0000497, 0.0000497)
7: (0.0082520, 0.0086916, 0.0082520, 0.0086916, -0.0001285, 0.0001285)
8: (0.0047755, 0.0050067, 0.0047755, 0.0050067, -0.0000676, 0.0000676)
9: (-0.0076694, -0.0074012, -0.0076694, -0.0074012, -0.0000784, 0.0000784)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.30 + 1.23 = 2.52 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0001178, upper bound: 0.0001178
