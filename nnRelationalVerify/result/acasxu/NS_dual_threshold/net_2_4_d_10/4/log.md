## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 269.4935248566139


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-73.5307999, 307.5443115, -73.5307999, 307.5443115, -381.0751038, 381.0751038)
1: (-51.2410049, 180.4736023, -51.2410049, 180.4736023, -231.7145996, 231.7145996)
2: (-27.6388569, 166.3133850, -27.6388569, 166.3133850, -193.9522400, 193.9522400)
3: (-37.8486443, 244.7528687, -37.8486443, 244.7528687, -282.6015015, 282.6014709)
4: (-51.2752876, 202.1299896, -51.2752876, 202.1299896, -253.4052734, 253.4052734)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.81 + 2.49 = 4.29 seconds
status: Status.VERIFIED
relational distance
Output dim: 4, lower bound: -226.9969763, upper bound: 226.9969763
