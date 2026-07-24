## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01246608


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986)
1: (-0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912)
2: (0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394)
3: (-0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711)
4: (-0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0039527, 0.0039527)
5: (-0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973)
6: (-0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037)
7: (-0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0221395, 0.0221395)
8: (0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723)
9: (-0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0153951, 0.0153951)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.02 + 2.39 = 3.42 seconds
status: Status.VERIFIED
relational distance
Output dim: 8, lower bound: -0.0138512, upper bound: 0.0138512
