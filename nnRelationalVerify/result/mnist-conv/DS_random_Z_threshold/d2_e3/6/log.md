## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.28670084832754067


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.6885729, -9.0492439, -10.6885729, -9.0492439, -0.8104372, 0.8104372)
1: (-11.3306236, -9.5733223, -11.3306236, -9.5733223, -0.9530272, 0.9530272)
2: (-10.4071865, -9.3814163, -10.4071865, -9.3814163, -0.6614900, 0.6614895)
3: (-10.2952652, -8.8535690, -10.2952652, -8.8535690, -0.8860140, 0.8860140)
4: (-11.9580545, -10.3808136, -11.9580545, -10.3808136, -0.6359582, 0.6359580)
5: (10.4790716, 11.3797112, 10.4790716, 11.3797112, -0.4973323, 0.4973323)
6: (-7.9649105, -6.4730420, -7.9649105, -6.4730420, -0.5911369, 0.5911371)
7: (-9.2039967, -7.8404670, -9.2039967, -7.8404670, -0.6536403, 0.6536403)
8: (-1.1489453, -0.3542111, -1.1489453, -0.3542111, -0.5198848, 0.5198848)
9: (-6.4444370, -5.5112157, -6.4444370, -5.5112157, -0.4426579, 0.4426579)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.56 + 34.48 = 59.05 seconds
status: Status.VERIFIED
relational distance
Output dim: 5, lower bound: -0.2460822, upper bound: 0.2460830
