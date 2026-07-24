## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 106.03367695430312


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-41.5758476, 52.7952728, -41.5758476, 52.7952728, -94.3711243, 94.3711166)
1: (-5.0236068, 3.0597908, -5.0236068, 3.0597908, -8.0833979, 8.0833969)
2: (-7.3437757, 9.9584923, -7.3437757, 9.9584923, -17.3022671, 17.3022690)
3: (-8.9331188, 6.6995373, -8.9331188, 6.6995373, -15.6326551, 15.6326561)
4: (-6.6065092, 7.7995100, -6.6065092, 7.7995100, -14.4060183, 14.4060192)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.99 + 1.19 = 2.17 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -84.0211374, upper bound: 84.0211373
