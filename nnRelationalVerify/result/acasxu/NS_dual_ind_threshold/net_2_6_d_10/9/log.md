## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 620.7657790946307


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-203.9254913, 371.2291565, -203.9254913, 371.2291565, -575.1546021, 575.1545410)
1: (-38.1354752, 47.1598244, -38.1354752, 47.1598244, -85.2953033, 85.2953033)
2: (-28.1897583, 61.3992081, -28.1897583, 61.3992081, -89.5889664, 89.5889664)
3: (-28.1089630, 81.1672745, -28.1089630, 81.1672745, -109.2762375, 109.2762375)
4: (-24.2144241, 76.7088699, -24.2144241, 76.7088699, -100.9232864, 100.9232864)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.46 + 1.58 = 4.04 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -491.2883155, upper bound: 491.2883150
