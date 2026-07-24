## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.11577233465117809


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.4019370, -3.9284534, -4.4019370, -3.9284534, -0.2124066, 0.2124066)
1: (-13.7760210, -13.2130232, -13.7760210, -13.2130232, -0.2293844, 0.2293844)
2: (-3.7660859, -3.3352268, -3.7660859, -3.3352268, -0.1993303, 0.1993306)
3: (-8.0845842, -7.4803863, -8.0845842, -7.4803863, -0.2368381, 0.2368381)
4: (-4.9877958, -4.4037080, -4.9877958, -4.4037080, -0.1964378, 0.1964378)
5: (-7.5293589, -7.0517821, -7.5293589, -7.0517821, -0.2042186, 0.2042186)
6: (-9.1943340, -8.7262716, -9.1943340, -8.7262716, -0.1798548, 0.1798549)
7: (-3.3216152, -2.8570738, -3.3216152, -2.8570738, -0.1720836, 0.1720836)
8: (-6.3617554, -5.8134308, -6.3617554, -5.8134308, -0.2266333, 0.2266333)
9: (4.9336762, 5.2771044, 4.9336762, 5.2771044, -0.1200635, 0.1200634)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.74 + 32.86 = 56.60 seconds
status: Status.VERIFIED
relational distance
Output dim: 9, lower bound: -0.0922734, upper bound: 0.0922735
