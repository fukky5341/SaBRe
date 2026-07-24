## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 27.40893787016518


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.4136937, 9.1970081, -2.4136937, 9.1970081, -11.6106987, 11.6106997)
1: (-6.7879753, 19.4114361, -6.7879753, 19.4114361, -26.1994114, 26.1994114)
2: (-9.7616444, 17.4784870, -9.7616444, 17.4784870, -27.2401276, 27.2401276)
3: (-5.6393709, 22.4973812, -5.6393709, 22.4973812, -28.1367493, 28.1367493)
4: (-8.9105253, 15.5090075, -8.9105253, 15.5090075, -24.4195328, 24.4195328)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.05 + 1.80 = 2.85 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -22.9523294, upper bound: 22.9523294
