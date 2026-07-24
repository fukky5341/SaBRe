## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.6880901157041572


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.2982141, 0.3126433, -0.2982141, 0.3126433, -0.6108574, 0.6108574)
1: (-0.0468057, 0.0059762, -0.0468057, 0.0059762, -0.0527820, 0.0527820)
2: (0.0042102, 0.0725880, 0.0042102, 0.0725880, -0.0683777, 0.0683777)
3: (-0.0210321, 0.0223050, -0.0210321, 0.0223050, -0.0433371, 0.0433371)
4: (0.0129541, 0.0779929, 0.0129541, 0.0779929, -0.0650388, 0.0650388)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.93 + 0.73 = 1.67 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.5602388, upper bound: 0.5602388
