## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.39004077726112174


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.7317963, -5.6235237, -6.7317963, -5.6235237, -0.8754816, 0.8754811)
1: (-9.6285591, -8.8130226, -9.6285591, -8.8130226, -0.4105911, 0.4105911)
2: (-5.6420774, -4.9479575, -5.6420774, -4.9479575, -0.4130940, 0.4130940)
3: (-6.0601602, -5.3838019, -6.0601602, -5.3838019, -0.5524635, 0.5524635)
4: (-6.7961836, -5.9421892, -6.7961836, -5.9421892, -0.4746804, 0.4746805)
5: (-2.9840603, -2.0013964, -2.9840603, -2.0013964, -0.5414741, 0.5414741)
6: (-6.1777620, -5.3024225, -6.1777620, -5.3024225, -0.6217980, 0.6217980)
7: (-8.3863106, -7.5939360, -8.3863106, -7.5939360, -0.4676871, 0.4676871)
8: (6.6702204, 7.4477329, 6.6702204, 7.4477329, -0.6688175, 0.6688175)
9: (-4.7721806, -3.9015856, -4.7721806, -3.9015856, -0.5672731, 0.5672729)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.82 + 33.30 = 56.12 seconds
status: Status.VERIFIED
relational distance
Output dim: 8, lower bound: -0.3501418, upper bound: 0.3501421
