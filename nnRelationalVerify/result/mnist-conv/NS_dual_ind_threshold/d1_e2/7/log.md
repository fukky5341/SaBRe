## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.19402795103245454


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.0046034, -4.2589965, -5.0046034, -4.2589965, -0.3481405, 0.3481405)
1: (-17.0620804, -16.3947487, -17.0620804, -16.3947487, -0.2912188, 0.2912188)
2: (4.6545324, 5.1439810, 4.6545324, 5.1439810, -0.2491665, 0.2491665)
3: (-2.1615372, -1.6257758, -2.1615372, -1.6257758, -0.2729695, 0.2729696)
4: (-12.4629517, -11.8319874, -12.4629517, -11.8319874, -0.4313741, 0.4313745)
5: (-6.6496477, -6.1696053, -6.6496477, -6.1696053, -0.2467099, 0.2467098)
6: (-5.6024122, -4.9923906, -5.6024122, -4.9923906, -0.3820338, 0.3820338)
7: (-6.0786829, -5.5656729, -6.0786829, -5.5656729, -0.2671933, 0.2671933)
8: (-1.6722612, -1.1792164, -1.6722612, -1.1792164, -0.2913177, 0.2913177)
9: (-5.9367647, -5.4333868, -5.9367647, -5.4333868, -0.2974308, 0.2974308)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 20.81 + 33.88 = 54.69 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.1607698, upper bound: 0.1607698
