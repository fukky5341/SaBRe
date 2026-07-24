## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 103.05292561237502


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-289.5221252, 388.0848694, -289.5221252, 388.0848694, -677.6069946, 677.6069946)
1: (-36.0558968, 22.0691967, -36.0558968, 22.0691967, -58.1250916, 58.1250839)
2: (-49.9897995, 68.7424545, -49.9897995, 68.7424545, -118.7322464, 118.7322464)
3: (-57.0502586, 47.5934982, -57.0502586, 47.5934982, -104.6437531, 104.6437531)
4: (-43.2357635, 54.9677200, -43.2357635, 54.9677200, -98.2034683, 98.2034760)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.79 + 1.58 = 4.37 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -91.0817009, upper bound: 91.0817009
