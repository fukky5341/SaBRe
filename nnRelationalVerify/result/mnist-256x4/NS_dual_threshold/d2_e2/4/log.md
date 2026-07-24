## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.045817377500000006


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9380678, 0.9917637, 0.9380678, 0.9917637, -0.0536959, 0.0536959)
1: (-0.0115945, -0.0026022, -0.0115945, -0.0026022, -0.0089923, 0.0089923)
2: (0.0075203, 0.0204642, 0.0075203, 0.0204642, -0.0129439, 0.0129439)
3: (-0.0090238, 0.0055741, -0.0090238, 0.0055741, -0.0145979, 0.0145979)
4: (-0.0041402, 0.0057734, -0.0041402, 0.0057734, -0.0099136, 0.0099136)
5: (0.0084180, 0.0549622, 0.0084180, 0.0549622, -0.0465442, 0.0465442)
6: (-0.0082169, 0.0051646, -0.0082169, 0.0051646, -0.0133815, 0.0133815)
7: (-0.0184287, -0.0037830, -0.0184287, -0.0037830, -0.0146457, 0.0146457)
8: (-0.0084818, 0.0135686, -0.0084818, 0.0135686, -0.0220504, 0.0220504)
9: (-0.0001674, 0.0124900, -0.0001674, 0.0124900, -0.0126574, 0.0126574)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.67 + 2.19 = 3.87 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0495323, upper bound: 0.0495323
