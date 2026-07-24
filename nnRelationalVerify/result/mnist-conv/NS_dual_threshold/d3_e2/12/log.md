## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.607673347343442


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.0849304, -6.3374538, -8.0849304, -6.3374538, -1.1709118, 1.1709118)
1: (-10.6706619, -9.0810947, -10.6706619, -9.0810947, -1.1206963, 1.1206963)
2: (-6.0996137, -4.4675536, -6.0996137, -4.4675536, -1.2553287, 1.2553287)
3: (-5.1394973, -3.5636799, -5.1394973, -3.5636799, -1.0224512, 1.0224514)
4: (-10.5023556, -8.8054028, -10.5023556, -8.8054028, -1.2936535, 1.2936535)
5: (-5.3703828, -4.0390539, -5.3703828, -4.0390539, -0.9127483, 0.9127483)
6: (-2.1165612, -0.4279108, -2.1165612, -0.4279108, -1.1403151, 1.1403148)
7: (-9.3995123, -7.9291315, -9.3995123, -7.9291315, -1.0946503, 1.0946503)
8: (7.2357993, 8.2458429, 7.2357993, 8.2458429, -0.8026900, 0.8026900)
9: (-5.9512353, -4.5768113, -5.9512353, -4.5768113, -0.8627329, 0.8627329)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.89 + 41.20 = 65.09 seconds
status: Status.VERIFIED
relational distance
Output dim: 8, lower bound: -0.5121318, upper bound: 0.5121317
