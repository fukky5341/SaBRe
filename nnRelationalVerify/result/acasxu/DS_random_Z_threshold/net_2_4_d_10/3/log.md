## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.1315312898389118


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0263298, 0.1585991, 0.0263298, 0.1585991, -0.1322694, 0.1322694)
1: (-0.0578788, 0.0968443, -0.0578788, 0.0968443, -0.1547231, 0.1547231)
2: (0.0255683, 0.1185939, 0.0255683, 0.1185939, -0.0930255, 0.0930255)
3: (-0.0280455, 0.1252133, -0.0280455, 0.1252133, -0.1532588, 0.1532588)
4: (0.0176807, 0.1139293, 0.0176807, 0.1139293, -0.0962487, 0.0962487)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.85 + 0.75 = 1.60 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.1156701, upper bound: 0.1156701
