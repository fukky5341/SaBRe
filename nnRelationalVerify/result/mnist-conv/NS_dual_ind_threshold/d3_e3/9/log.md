## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.8863165295423633


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.6904020, -4.7620134, -6.6904020, -4.7620134, -1.6564002, 1.6564002)
1: (-6.9654446, -5.3930106, -6.9654446, -5.3930106, -1.4029658, 1.4029660)
2: (-5.7929034, -4.3129930, -5.7929034, -4.3129930, -1.1872196, 1.1872196)
3: (-5.6416736, -3.9028914, -5.6416736, -3.9028914, -1.4227171, 1.4227166)
4: (-6.1677923, -4.4485168, -6.1677923, -4.4485168, -1.6802793, 1.6802793)
5: (-6.2062693, -4.6347828, -6.2062693, -4.6347828, -1.3965995, 1.3965993)
6: (-11.0836697, -9.0797215, -11.0836697, -9.0797215, -1.6567426, 1.6567428)
7: (3.1360507, 4.5362215, 3.1360507, 4.5362215, -1.3662617, 1.3662620)
8: (-4.0897732, -2.4230666, -4.0897732, -2.4230666, -1.2649808, 1.2649810)
9: (-2.4739165, -1.3050532, -2.4739165, -1.3050532, -1.1688633, 1.1688633)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.99 + 33.29 = 56.28 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.7535166, upper bound: 0.7535159
