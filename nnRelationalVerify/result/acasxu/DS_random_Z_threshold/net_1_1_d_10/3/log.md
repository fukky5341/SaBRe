## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.8274327535446416


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017)
1: (-0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568)
2: (-0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153)
3: (-0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125)
4: (-0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.69 + 0.83 = 1.52 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.7126415, upper bound: 0.7126415
