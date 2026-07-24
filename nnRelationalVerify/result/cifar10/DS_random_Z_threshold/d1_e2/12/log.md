## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 12)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.010348451682571029


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.9781790, -3.3652067, -3.9781790, -3.3652067, -0.2058250, 0.2058250)
1: (-4.2116995, -3.2910633, -4.2116995, -3.2910633, -0.2775142, 0.2775142)
2: (-0.6417039, -0.4882156, -0.6417039, -0.4882156, -0.0167031, 0.0167031)
3: (-0.2540174, -0.0551738, -0.2540174, -0.0551738, -0.0638294, 0.0638294)
4: (-0.7684377, -0.4622545, -0.7684377, -0.4622545, -0.0469890, 0.0469890)
5: (-0.3594077, -0.0952139, -0.3594077, -0.0952139, -0.0544099, 0.0544099)
6: (-0.4537750, -0.1948292, -0.4537750, -0.1948292, -0.0458589, 0.0458589)
7: (-0.5558734, -0.3945007, -0.5558734, -0.3945007, -0.0225825, 0.0225825)
8: (-5.8459005, -5.1659765, -5.8459005, -5.1659765, -0.1950523, 0.1950523)
9: (-3.6860623, -3.1346951, -3.6860623, -3.1346951, -0.1700248, 0.1700249)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.03 + 19.58 = 27.61 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0083654, upper bound: 0.0083708
