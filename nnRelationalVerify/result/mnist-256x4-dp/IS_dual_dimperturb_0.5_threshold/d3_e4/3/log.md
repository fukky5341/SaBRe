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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=44, inp2_unstable=44, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.3598073, 1.0719240, 0.3598073, 1.0719240, -0.7121167, 0.7121167)
1: (-0.3086743, 0.3007838, -0.3086743, 0.3007838, -0.6094581, 0.6094581)
2: (-0.2211845, 0.4015892, -0.2211845, 0.4015892, -0.6227737, 0.6227737)
3: (-0.2247453, 0.3019670, -0.2247453, 0.3019670, -0.5267123, 0.5267123)
4: (-0.3198898, 0.2842786, -0.3198898, 0.2842786, -0.6041684, 0.6041684)
5: (-0.3446322, 0.4508855, -0.3446322, 0.4508855, -0.7955177, 0.7955177)
6: (-0.2470306, 0.3340175, -0.2470306, 0.3340175, -0.5810481, 0.5810481)
7: (-0.3327097, 0.3429792, -0.3327097, 0.3429792, -0.6756889, 0.6756889)
8: (-0.3203780, 0.4160657, -0.3203780, 0.4160657, -0.7364437, 0.7364437)
9: (-0.3137136, 0.4036881, -0.3137136, 0.4036881, -0.7174017, 0.7174017)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.71 + 2.41 = 4.12 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.5874821, upper bound: 0.5874822
