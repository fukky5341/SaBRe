## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 2814.44097722469


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-452.7791443, 1276.7998047, -452.7791443, 1276.7998047, -1729.5787354, 1729.5787354)
1: (-644.6803589, 1322.9897461, -644.6803589, 1322.9897461, -1967.6701660, 1967.6701660)
2: (-544.1079712, 1464.9575195, -544.1079712, 1464.9575195, -2009.0654297, 2009.0654297)
3: (-580.0340576, 1829.0605469, -580.0340576, 1829.0605469, -2409.0944824, 2409.0944824)
4: (-484.6902161, 1726.1364746, -484.6902161, 1726.1364746, -2210.8266602, 2210.8266602)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.71 + 2.52 = 3.23 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -2227.8615041, upper bound: 2227.8615041
