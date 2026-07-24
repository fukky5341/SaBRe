## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 6160.067995326712


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2608.1174316, 3153.5952148, -2608.1174316, 3153.5952148, -5761.7119141, 5761.7119141)
1: (-298.7129822, 266.1875305, -298.7129822, 266.1875305, -564.9003906, 564.9003906)
2: (-171.0293121, 299.1513672, -171.0293121, 299.1513672, -470.1806030, 470.1806030)
3: (-143.8485413, 309.5449524, -143.8485413, 309.5449524, -453.3934937, 453.3934937)
4: (-203.7647858, 265.6119385, -203.7647858, 265.6119385, -469.3767090, 469.3767090)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.52 + 2.00 = 4.52 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -5269.2046909, upper bound: 5269.2046909
