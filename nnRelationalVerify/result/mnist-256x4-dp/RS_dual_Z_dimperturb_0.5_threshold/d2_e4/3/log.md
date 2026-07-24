## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.40684923


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1947630, 0.2380742, -0.1947630, 0.2380742, -0.4328372, 0.4328372)
1: (0.6230491, 1.0690367, 0.6230491, 1.0690367, -0.4459876, 0.4459876)
2: (-0.1574410, 0.2053816, -0.1574410, 0.2053816, -0.3628225, 0.3628225)
3: (-0.1035686, 0.1519503, -0.1035686, 0.1519503, -0.2555190, 0.2555190)
4: (-0.1609858, 0.1617371, -0.1609858, 0.1617371, -0.3227229, 0.3227229)
5: (-0.1603857, 0.1834070, -0.1603857, 0.1834070, -0.3437927, 0.3437927)
6: (-0.2093027, 0.1913667, -0.2093027, 0.1913667, -0.4006695, 0.4006695)
7: (-0.1555940, 0.2059899, -0.1555940, 0.2059899, -0.3615839, 0.3615839)
8: (-0.1150274, 0.2745139, -0.1150274, 0.2745139, -0.3895413, 0.3895413)
9: (-0.1946198, 0.1923223, -0.1946198, 0.1923223, -0.3869421, 0.3869421)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 2.11 = 3.54 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.3918807, upper bound: 0.3918807
