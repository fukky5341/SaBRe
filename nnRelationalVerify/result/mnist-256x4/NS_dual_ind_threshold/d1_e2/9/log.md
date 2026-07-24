## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.005535986938780396


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0029655, -0.0006652, -0.0029655, -0.0006652, -0.0017114, 0.0017114)
1: (-0.0041270, 0.0028174, -0.0041270, 0.0028174, -0.0053524, 0.0053524)
2: (0.0032672, 0.0083234, 0.0032672, 0.0083234, -0.0039431, 0.0039431)
3: (-0.0038078, -0.0030785, -0.0038078, -0.0030785, -0.0005338, 0.0005338)
4: (0.0057481, 0.0075547, 0.0057481, 0.0075547, -0.0013508, 0.0013508)
5: (-0.0036517, -0.0013294, -0.0036517, -0.0013294, -0.0017003, 0.0017003)
6: (-0.0062734, -0.0056955, -0.0062734, -0.0056955, -0.0004414, 0.0004414)
7: (-0.0008896, 0.0022596, -0.0008896, 0.0022596, -0.0025239, 0.0025239)
8: (-0.0016782, -0.0008535, -0.0016782, -0.0008535, -0.0006134, 0.0006134)
9: (1.0042045, 1.0126595, 1.0042045, 1.0126595, -0.0064179, 0.0064179)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.41 + 2.00 = 3.40 seconds
status: Status.VERIFIED
relational distance
Output dim: 9, lower bound: -0.0046953, upper bound: 0.0046953
