## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 28.546024900262832


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-107.8605957, 136.0422363, -107.8605957, 136.0422363, -243.9028320, 243.9028168)
1: (-12.6449718, 5.3934770, -12.6449718, 5.3934770, -18.0384445, 18.0384445)
2: (-5.7488804, 14.7411308, -5.7488804, 14.7411308, -20.4900112, 20.4900093)
3: (-9.0719624, 18.1302338, -9.0719624, 18.1302338, -27.2021942, 27.2021942)
4: (-5.7443848, 15.0105000, -5.7443848, 15.0105000, -20.7548847, 20.7548847)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.11 + 1.49 = 4.60 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -22.5109366, upper bound: 22.5109366
