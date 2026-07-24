## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.130931364


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0132380, 0.0356402, -0.0132380, 0.0356402, -0.0488782, 0.0488782)
1: (-0.0171666, 0.0240945, -0.0171666, 0.0240945, -0.0412610, 0.0412610)
2: (-0.0284746, 0.0458213, -0.0284746, 0.0458213, -0.0742959, 0.0742959)
3: (-0.0403148, 0.0362702, -0.0403148, 0.0362702, -0.0765850, 0.0765850)
4: (-0.0301056, 0.0611707, -0.0301056, 0.0611707, -0.0912763, 0.0912763)
5: (-0.0831561, 0.0494512, -0.0831561, 0.0494512, -0.1326073, 0.1326073)
6: (-0.0274300, 0.0826313, -0.0274300, 0.0826313, -0.1100613, 0.1100613)
7: (-0.0414651, 0.0204140, -0.0414651, 0.0204140, -0.0618791, 0.0618791)
8: (0.8548078, 1.0175655, 0.8548078, 1.0175655, -0.1627577, 0.1627577)
9: (-0.0429216, 0.0591904, -0.0429216, 0.0591904, -0.1021120, 0.1021120)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.89 + 2.98 = 3.87 seconds
status: Status.VERIFIED
relational distance
Output dim: 8, lower bound: -0.1423167, upper bound: 0.1423167
