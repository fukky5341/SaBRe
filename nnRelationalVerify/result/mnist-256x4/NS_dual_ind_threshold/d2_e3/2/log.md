## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0157495625


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293)
1: (-0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382)
2: (0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210)
3: (-0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023)
4: (0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575)
5: (0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635)
6: (-0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247)
7: (-0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719)
8: (-0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123)
9: (-0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.75 + 2.20 = 3.94 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0179995, upper bound: 0.0179995
