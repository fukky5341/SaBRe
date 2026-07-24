## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 1101.5661254361103


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450)
1: (-228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977)
2: (-233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266)
3: (-369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684)
4: (-372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.84 + 2.13 = 2.98 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -853.3029743, upper bound: 853.3029743
