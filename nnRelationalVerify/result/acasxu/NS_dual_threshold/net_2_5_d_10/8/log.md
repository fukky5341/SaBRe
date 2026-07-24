## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 45.77727815756507


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-39.1516075, 71.9633865, -39.1516075, 71.9633865, -111.1149902, 111.1149902)
1: (-11.6254883, 21.8392792, -11.6254883, 21.8392792, -33.4647675, 33.4647675)
2: (-6.4613066, 24.5460224, -6.4613066, 24.5460224, -31.0073280, 31.0073280)
3: (-12.0124016, 26.7618332, -12.0124016, 26.7618332, -38.7742310, 38.7742310)
4: (-7.7486634, 25.6497765, -7.7486634, 25.6497765, -33.3984413, 33.3984413)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.13 + 1.76 = 3.90 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -36.1622943, upper bound: 36.1622943
