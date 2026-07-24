## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.07616917832846994


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.2973299, -5.7840595, -6.2973299, -5.7840595, -0.2682567, 0.2682567)
1: (-10.4731035, -9.9518127, -10.4731035, -9.9518127, -0.2702231, 0.2702231)
2: (-5.9708381, -5.5498676, -5.9708381, -5.5498676, -0.1779858, 0.1779858)
3: (-8.6906557, -8.0953903, -8.6906557, -8.0953903, -0.2962327, 0.2962327)
4: (-3.6191802, -3.2003231, -3.6191802, -3.2003231, -0.2130675, 0.2130674)
5: (-4.6469135, -4.1102047, -4.6469135, -4.1102047, -0.3580337, 0.3580337)
6: (-6.4201670, -5.8744335, -6.4201670, -5.8744335, -0.3354692, 0.3354688)
7: (-12.1619349, -11.6063690, -12.1619349, -11.6063690, -0.2064904, 0.2064903)
8: (-6.0446892, -5.4998188, -6.0446892, -5.4998188, -0.2214410, 0.2214410)
9: (2.8756719, 3.1103945, 2.8756719, 3.1103945, -0.0907785, 0.0907785)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 20.74 + 32.91 = 53.65 seconds
status: Status.VERIFIED
relational distance
Output dim: 9, lower bound: -0.0676346, upper bound: 0.0676342
