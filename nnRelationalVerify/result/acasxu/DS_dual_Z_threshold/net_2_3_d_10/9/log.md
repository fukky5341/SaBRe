## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 104.7398868834598


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-25.2493973, 78.3703461, -25.2493973, 78.3703461, -103.6197357, 103.6197433)
1: (-17.9429550, 53.7068787, -17.9429550, 53.7068787, -71.6498260, 71.6498184)
2: (-20.3127823, 48.6183968, -20.3127823, 48.6183968, -68.9311752, 68.9311752)
3: (-21.6872959, 65.0137177, -21.6872959, 65.0137177, -86.7010117, 86.7010117)
4: (-31.8455753, 53.3359947, -31.8455753, 53.3359947, -85.1815720, 85.1815720)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.66 + 1.56 = 3.21 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -93.1207181, upper bound: 93.1207181
