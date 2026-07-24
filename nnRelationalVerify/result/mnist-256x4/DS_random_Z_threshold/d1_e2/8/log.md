## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.04879385138341862


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1646274, 0.0414454, -0.1646274, 0.0414454, -0.2060728, 0.2060728)
1: (-0.0118636, 0.0338192, -0.0118636, 0.0338192, -0.0447405, 0.0447405)
2: (-0.0226597, 0.2007567, -0.0226597, 0.2007567, -0.2219808, 0.2219808)
3: (0.9622606, 1.0222970, 0.9622606, 1.0222970, -0.0600364, 0.0600364)
4: (-0.0086431, 0.0185173, -0.0086431, 0.0185173, -0.0271604, 0.0271604)
5: (-0.0321408, 0.0176856, -0.0321408, 0.0176856, -0.0498264, 0.0498264)
6: (-0.0100758, 0.0379788, -0.0100758, 0.0379788, -0.0480546, 0.0480546)
7: (-0.0680835, 0.0001831, -0.0680835, 0.0001831, -0.0682666, 0.0682666)
8: (-0.0314547, 0.0133356, -0.0314547, 0.0133356, -0.0447903, 0.0447903)
9: (-0.0541051, 0.0137480, -0.0541051, 0.0137480, -0.0678531, 0.0678531)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.95 + 1.88 = 2.82 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -0.0437945, upper bound: 0.0437945
