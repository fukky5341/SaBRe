## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 88.51082491230967


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-26.1755524, 93.5286026, -26.1755524, 93.5286026, -119.7041473, 119.7041473)
1: (-16.2761669, 57.2903519, -16.2761669, 57.2903519, -73.5665207, 73.5665207)
2: (-9.0787802, 52.5494881, -9.0787802, 52.5494881, -61.6282654, 61.6282616)
3: (-12.7195129, 78.0047607, -12.7195129, 78.0047607, -90.7242737, 90.7242737)
4: (-16.6739655, 63.5641556, -16.6739655, 63.5641556, -80.2381210, 80.2381210)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.89 + 1.63 = 3.52 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -78.7537488, upper bound: 78.7537488
