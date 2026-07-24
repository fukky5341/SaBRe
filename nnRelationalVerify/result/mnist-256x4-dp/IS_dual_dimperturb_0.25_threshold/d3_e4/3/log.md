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
Threshold: 0.03662127


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9630512, 0.9910084, 0.9630512, 0.9910084, -0.0279572, 0.0279572)
1: (-0.0048267, -0.0031470, -0.0048267, -0.0031470, -0.0016797, 0.0016797)
2: (0.0085175, 0.0154349, 0.0085175, 0.0154349, -0.0066937, 0.0066937)
3: (-0.0085420, -0.0051500, -0.0085420, -0.0051500, -0.0033921, 0.0033921)
4: (0.0021765, 0.0049898, 0.0021765, 0.0049898, -0.0026994, 0.0026994)
5: (0.0096723, 0.0338914, 0.0096723, 0.0338914, -0.0242191, 0.0242191)
6: (-0.0030935, -0.0002774, -0.0030935, -0.0002774, -0.0028161, 0.0028161)
7: (-0.0110790, -0.0055027, -0.0110790, -0.0055027, -0.0055763, 0.0055763)
8: (-0.0053595, 0.0031316, -0.0053595, 0.0031316, -0.0084912, 0.0084912)
9: (0.0009863, 0.0048139, 0.0009863, 0.0048139, -0.0038276, 0.0038276)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.80 + 1.74 = 3.54 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0247646, upper bound: 0.0247646
