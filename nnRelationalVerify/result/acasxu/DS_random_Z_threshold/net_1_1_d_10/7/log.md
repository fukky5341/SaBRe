## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 974.2101457357207


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-204.6976013, 696.8228149, -204.6976013, 696.8228149, -901.5203857, 901.5203857)
1: (-334.7795410, 851.1083374, -334.7795410, 851.1083374, -1185.8878174, 1185.8878174)
2: (-233.0683441, 900.9995117, -233.0683441, 900.9995117, -1134.0678711, 1134.0678711)
3: (-594.6608887, 866.9309082, -594.6608887, 866.9309082, -1461.5917969, 1461.5917969)
4: (-370.8992920, 924.7813721, -370.8992920, 924.7813721, -1295.6806641, 1295.6806641)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.79 + 1.94 = 2.72 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -843.2115492, upper bound: 843.2115492
