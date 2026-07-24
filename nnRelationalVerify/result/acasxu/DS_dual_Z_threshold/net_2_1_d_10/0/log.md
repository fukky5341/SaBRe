## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 1937.0157959649773


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461)
1: (-576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391)
2: (-492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311)
3: (-691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342)
4: (-652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.24 + 2.12 = 3.36 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -1541.9488800, upper bound: 1541.9488800
