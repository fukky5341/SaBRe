## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.023662612462354928


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0094807, 0.0359102, 0.0094807, 0.0359102, -0.0264296, 0.0264296)
1: (-0.0260700, -0.0163090, -0.0260700, -0.0163090, -0.0097610, 0.0097610)
2: (0.0123426, 0.0250938, 0.0123426, 0.0250938, -0.0127512, 0.0127512)
3: (-0.0197087, -0.0123577, -0.0197087, -0.0123577, -0.0073510, 0.0073510)
4: (0.0171183, 0.0239822, 0.0171183, 0.0239822, -0.0068639, 0.0068639)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.27 + 0.73 = 3.00 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0197676, upper bound: 0.0197676
