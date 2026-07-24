## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 15.545989091570618


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311)
1: (-5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291)
2: (-7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632)
3: (-2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622)
4: (-10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.55 + 1.03 = 1.58 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -12.2720720, upper bound: 12.2720720
