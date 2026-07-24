## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00199528


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000766, 0.0000766)
1: (-0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0028695, 0.0028695)
2: (0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0034436, 0.0034436)
3: (-0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0253993, 0.0253993)
4: (-0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0019318, 0.0019318)
5: (0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0019524, 0.0019524)
6: (0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009496, 0.0009496)
7: (-0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0065824, 0.0065824)
8: (0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0052222, 0.0052222)
9: (0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0093926, 0.0093926)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.62 + 2.64 = 5.26 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0024941, upper bound: 0.0024941
