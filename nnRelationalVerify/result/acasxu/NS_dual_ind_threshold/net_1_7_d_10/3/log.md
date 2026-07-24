## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.004122329976761439


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0208230, -0.0165418, -0.0208230, -0.0165418, -0.0042812, 0.0042812)
1: (-0.0188711, -0.0003911, -0.0188711, -0.0003911, -0.0184799, 0.0184799)
2: (-0.0265466, -0.0127214, -0.0265466, -0.0127214, -0.0138252, 0.0138252)
3: (-0.0170388, 0.0011029, -0.0170388, 0.0011029, -0.0181416, 0.0181416)
4: (-0.0230370, -0.0110406, -0.0230370, -0.0110406, -0.0119964, 0.0119964)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.90 + 0.60 = 1.50 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0032806, upper bound: 0.0032806
