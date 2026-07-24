## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 2.1406068709517623


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340)
1: (-0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480)
2: (-1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007)
3: (-1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638)
4: (-2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.67 + 1.07 = 2.74 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -1.9315851, upper bound: 1.9315851
