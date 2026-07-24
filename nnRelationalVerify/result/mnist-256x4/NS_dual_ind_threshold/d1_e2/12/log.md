## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.006079831986192889


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0035758, -0.0012563, -0.0035758, -0.0012563, -0.0023194, 0.0023194)
1: (0.0016244, 0.0081726, 0.0016244, 0.0081726, -0.0064150, 0.0064150)
2: (0.0083484, 0.0138261, 0.0083484, 0.0138261, -0.0048904, 0.0048904)
3: (-0.0028812, -0.0016663, -0.0028812, -0.0016663, -0.0011140, 0.0011140)
4: (0.0051316, 0.0069282, 0.0051316, 0.0069282, -0.0016154, 0.0016154)
5: (-0.0044022, -0.0018830, -0.0044022, -0.0018830, -0.0025192, 0.0025192)
6: (-0.0058725, -0.0053187, -0.0058725, -0.0053187, -0.0005538, 0.0005538)
7: (-0.0056295, -0.0029628, -0.0056295, -0.0029628, -0.0026141, 0.0026141)
8: (-0.0038946, -0.0025554, -0.0038946, -0.0025554, -0.0013392, 0.0013392)
9: (0.9993659, 1.0075498, 0.9993659, 1.0075498, -0.0077749, 0.0077749)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.25 + 2.75 = 4.01 seconds
status: Status.VERIFIED
relational distance
Output dim: 9, lower bound: -0.0054892, upper bound: 0.0054892
