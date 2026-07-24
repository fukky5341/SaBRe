## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.06674455


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125)
1: (-0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677)
2: (-0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924)
3: (0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574)
4: (-0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169)
5: (-0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054)
6: (-0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380)
7: (-0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123)
8: (-0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561)
9: (-0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.45 + 2.99 = 4.44 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -0.0785230, upper bound: 0.0785227
