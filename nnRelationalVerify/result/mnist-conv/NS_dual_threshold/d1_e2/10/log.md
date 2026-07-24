## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.16089163390612288


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.6821756, -8.0933456, -8.6821756, -8.0933456, -0.2770245, 0.2770245)
1: (-7.1159964, -6.6619458, -7.1159964, -6.6619458, -0.2561193, 0.2561193)
2: (4.5109024, 4.9215412, 4.5109024, 4.9215412, -0.2134030, 0.2134030)
3: (-4.2848811, -3.7612104, -4.2848811, -3.7612104, -0.2975709, 0.2975707)
4: (-14.2449722, -13.6650887, -14.2449722, -13.6650887, -0.2695510, 0.2695512)
5: (-12.1641245, -11.6924572, -12.1641245, -11.6924572, -0.1736450, 0.1736450)
6: (-9.5596533, -9.0381107, -9.5596533, -9.0381107, -0.2057003, 0.2057003)
7: (-4.0826044, -3.5887451, -4.0826044, -3.5887451, -0.1782740, 0.1782740)
8: (1.9667611, 2.2375932, 1.9667611, 2.2375932, -0.1004693, 0.1004693)
9: (-8.4021912, -7.9056449, -8.4021912, -7.9056449, -0.2664070, 0.2664068)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.71 + 32.84 = 54.54 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.1329896, upper bound: 0.1329896
