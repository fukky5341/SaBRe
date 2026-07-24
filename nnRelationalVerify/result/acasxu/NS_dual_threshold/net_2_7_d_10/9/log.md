## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 3066.328627295288


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1203.1627197, 1532.7302246, -1203.1627197, 1532.7302246, -2735.8920898, 2735.8920898)
1: (-143.6590271, 126.9177017, -143.6590271, 126.9177017, -270.5766602, 270.5766602)
2: (-82.0869064, 139.7667542, -82.0869064, 139.7667542, -221.8536682, 221.8536682)
3: (-79.8995972, 147.3026733, -79.8995972, 147.3026733, -227.2022552, 227.2022552)
4: (-95.7516632, 123.8205185, -95.7516632, 123.8205185, -219.5721741, 219.5721741)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.60 + 1.78 = 4.38 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -2471.7856104, upper bound: 2471.7856104
