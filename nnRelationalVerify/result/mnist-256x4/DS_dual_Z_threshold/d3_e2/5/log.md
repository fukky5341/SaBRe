## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.140417184


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0941507, 0.0756852, -0.0941507, 0.0756852, -0.1698359, 0.1698359)
1: (-0.0573000, 0.0852287, -0.0573000, 0.0852287, -0.1425287, 0.1425287)
2: (-0.1352168, 0.0655241, -0.1352168, 0.0655241, -0.2007408, 0.2007408)
3: (0.8851405, 1.0497663, 0.8851405, 1.0497663, -0.1646258, 0.1646258)
4: (-0.0272898, 0.1162062, -0.0272898, 0.1162062, -0.1434960, 0.1434960)
5: (-0.0489096, 0.2939964, -0.0489096, 0.2939964, -0.3429060, 0.3429060)
6: (-0.1204223, 0.0862378, -0.1204223, 0.0862378, -0.2066601, 0.2066601)
7: (-0.1441980, 0.0118736, -0.1441980, 0.0118736, -0.1560716, 0.1560716)
8: (-0.0537071, 0.1002814, -0.0537071, 0.1002814, -0.1539885, 0.1539885)
9: (-0.1409980, 0.0980089, -0.1409980, 0.0980089, -0.2390069, 0.2390069)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.15 + 4.21 = 6.36 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -0.1462679, upper bound: 0.1462679
