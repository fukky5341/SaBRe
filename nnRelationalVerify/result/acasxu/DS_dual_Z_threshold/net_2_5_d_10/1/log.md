## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 4409.221102806015


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1750.0124512, 3321.1867676, -1750.0124512, 3321.1867676, -5071.1992188, 5071.1992188)
1: (-591.4364624, 1249.1158447, -591.4364624, 1249.1158447, -1840.5522461, 1840.5522461)
2: (-302.9573364, 1257.5653076, -302.9573364, 1257.5653076, -1560.5225830, 1560.5225830)
3: (-690.8545532, 1531.0914307, -690.8545532, 1531.0914307, -2221.9460449, 2221.9460449)
4: (-393.2388916, 1293.9045410, -393.2388916, 1293.9045410, -1687.1434326, 1687.1434326)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.04 + 2.03 = 4.07 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -3656.1923533, upper bound: 3656.1923533
