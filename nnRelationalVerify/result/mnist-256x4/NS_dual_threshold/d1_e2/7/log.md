## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0036733774027241837


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040893, -0.0033038, -0.0040893, -0.0033038, -0.0007855, 0.0007855)
1: (-0.0058011, -0.0038806, -0.0058011, -0.0038806, -0.0019206, 0.0019206)
2: (0.9677920, 0.9712254, 0.9677920, 0.9712254, -0.0034334, 0.0034334)
3: (0.0213566, 0.0340701, 0.0213566, 0.0340701, -0.0092668, 0.0092668)
4: (-0.0032843, -0.0017444, -0.0032843, -0.0017444, -0.0015399, 0.0015399)
5: (0.0133494, 0.0149283, 0.0133494, 0.0149283, -0.0015789, 0.0015789)
6: (0.0037932, 0.0051210, 0.0037932, 0.0051210, -0.0013278, 0.0013278)
7: (-0.0166078, -0.0133130, -0.0166078, -0.0133130, -0.0032948, 0.0032948)
8: (0.0035533, 0.0061672, 0.0035533, 0.0061672, -0.0026139, 0.0026139)
9: (0.0035850, 0.0088170, 0.0035850, 0.0088170, -0.0052320, 0.0052320)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.54 + 1.75 = 3.29 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0030886, upper bound: 0.0030886
