## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.195173894


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0394248, 0.0546533, -0.0394248, 0.0546533, -0.0940781, 0.0940781)
1: (-0.0606665, 0.1004384, -0.0606665, 0.1004384, -0.1611049, 0.1611049)
2: (-0.0892648, 0.1406264, -0.0892648, 0.1406264, -0.2298912, 0.2298912)
3: (-0.0573224, 0.0218208, -0.0573224, 0.0218208, -0.0791432, 0.0791432)
4: (-0.0867973, 0.0994622, -0.0867973, 0.0994622, -0.1862595, 0.1862595)
5: (-0.0849066, 0.0887389, -0.0849066, 0.0887389, -0.1736455, 0.1736455)
6: (0.7925102, 1.0225793, 0.7925102, 1.0225793, -0.2300692, 0.2300692)
7: (-0.1114460, 0.0865751, -0.1114460, 0.0865751, -0.1980211, 0.1980211)
8: (-0.0711323, 0.1367783, -0.0711323, 0.1367783, -0.2079106, 0.2079106)
9: (-0.0716718, 0.0720556, -0.0716718, 0.0720556, -0.1437274, 0.1437274)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.42 + 3.11 = 5.53 seconds
status: Status.VERIFIED
relational distance
Output dim: 6, lower bound: -0.2012103, upper bound: 0.2012103
