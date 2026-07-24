## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.017462771472107105


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0053368, -0.0039028, -0.0053368, -0.0039028, -0.0012668, 0.0012668)
1: (0.0013577, 0.0092979, 0.0013577, 0.0092979, -0.0070140, 0.0070140)
2: (-0.0058064, 0.0119329, -0.0058064, 0.0119329, -0.0156700, 0.0156700)
3: (0.0023058, 0.0097812, 0.0023058, 0.0097812, -0.0066034, 0.0066034)
4: (1.0056959, 1.0346975, 1.0056959, 1.0346975, -0.0256185, 0.0256185)
5: (0.0033495, 0.0089915, 0.0033495, 0.0089915, -0.0049838, 0.0049838)
6: (-0.0174440, -0.0101018, -0.0174440, -0.0101018, -0.0064857, 0.0064857)
7: (-0.0110285, -0.0100919, -0.0110285, -0.0100919, -0.0008273, 0.0008273)
8: (-0.0038746, 0.0011983, -0.0038746, 0.0011983, -0.0044811, 0.0044811)
9: (-0.0241700, 0.0012260, -0.0241700, 0.0012260, -0.0224335, 0.0224335)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.89 + 2.26 = 4.15 seconds
status: Status.VERIFIED
relational distance
Output dim: 4, lower bound: -0.0156673, upper bound: 0.0156673
