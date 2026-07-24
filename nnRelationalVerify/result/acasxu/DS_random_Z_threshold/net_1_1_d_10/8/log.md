## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 142.08781015972576


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448)
1: (-24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693)
2: (-20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918)
3: (-40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481)
4: (-30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.79 + 1.64 = 2.43 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -112.6516518, upper bound: 112.6516518
