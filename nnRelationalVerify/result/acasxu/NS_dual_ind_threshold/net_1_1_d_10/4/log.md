## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 105.73809588791504


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396)
1: (-25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411)
2: (-22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495)
3: (-42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475)
4: (-33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.65 + 1.54 = 2.19 seconds
status: Status.VERIFIED
relational distance
Output dim: 4, lower bound: -95.4385641, upper bound: 95.4385641
