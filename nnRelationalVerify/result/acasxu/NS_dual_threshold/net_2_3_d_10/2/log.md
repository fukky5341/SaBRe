## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 1.997507971622052


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.5309063, 1.3496586, -0.5309063, 1.3496586, -1.8805649, 1.8805649)
1: (-0.4135189, 0.8432285, -0.4135189, 0.8432285, -1.2567474, 1.2567474)
2: (-0.3319868, 0.9570781, -0.3319868, 0.9570781, -1.2890649, 1.2890649)
3: (-0.3423531, 1.0881081, -0.3423531, 1.0881081, -1.4304612, 1.4304612)
4: (-0.5717137, 0.9521344, -0.5717137, 0.9521344, -1.5238481, 1.5238481)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.68 + 1.03 = 2.71 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -1.5816230, upper bound: 1.5816230
