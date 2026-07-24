## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.15539725532349155


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.5937271, -7.8297682, -8.5937271, -7.8297682, -0.3359882, 0.3359880)
1: (3.2346950, 3.9044790, 3.2346950, 3.9044790, -0.2471337, 0.2471337)
2: (-4.6458235, -4.0411386, -4.6458235, -4.0411386, -0.2421845, 0.2421845)
3: (-12.6903105, -11.9918861, -12.6903105, -11.9918861, -0.2825890, 0.2825888)
4: (-5.8708782, -5.1399159, -5.8708782, -5.1399159, -0.2429190, 0.2429190)
5: (-9.0187454, -8.2395420, -9.0187454, -8.2395420, -0.3344002, 0.3344002)
6: (-5.8345094, -5.1275363, -5.8345094, -5.1275363, -0.2929747, 0.2929749)
7: (-4.5132179, -4.0134449, -4.5132179, -4.0134449, -0.2598656, 0.2598656)
8: (-2.4657760, -1.8733640, -2.4657760, -1.8733640, -0.2289033, 0.2289033)
9: (-12.1005459, -11.3783417, -12.1005459, -11.3783417, -0.3053659, 0.3053659)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.46 + 34.08 = 57.55 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.1209252, upper bound: 0.1209250
