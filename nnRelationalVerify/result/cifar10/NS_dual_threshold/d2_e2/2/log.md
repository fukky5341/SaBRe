## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 2)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.1172187745920931


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.4231331, 2.3679526, 1.4231331, 2.3679526, -0.7208447, 0.7208447)
1: (-2.7800131, -0.6669235, -2.7800131, -0.6669235, -1.1597719, 1.1597719)
2: (-0.9595081, -0.2979927, -0.9595081, -0.2979927, -0.1820599, 0.1820599)
3: (-0.7473869, -0.2034371, -0.7473869, -0.2034371, -0.4440773, 0.4440773)
4: (-2.4312544, -1.8530471, -2.4312544, -1.8530471, -0.3017851, 0.3017851)
5: (-2.3475487, -1.5588803, -2.3475487, -1.5588803, -0.5187949, 0.5187950)
6: (-1.0046231, 0.0270669, -1.0046231, 0.0270669, -0.6226485, 0.6226487)
7: (-2.7474337, -1.3944523, -2.7474337, -1.3944523, -1.0512700, 1.0512698)
8: (-2.5697012, -1.1055967, -2.5697012, -1.1055967, -0.6966766, 0.6966767)
9: (-2.8024588, -1.2105114, -2.8024588, -1.2105114, -0.6293665, 0.6293665)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.40 + 34.33 = 41.74 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -0.0923400, upper bound: 0.0923441
