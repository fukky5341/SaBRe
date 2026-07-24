## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.9713820178729172


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.1073970, 1.1076047, 0.1073970, 1.1076047, -1.0002078, 1.0002078)
1: (-0.4708654, 0.4723013, -0.4708654, 0.4723013, -0.9431667, 0.9431667)
2: (-0.3621331, 0.5750049, -0.3621331, 0.5750049, -0.9371381, 0.9371381)
3: (-0.3326984, 0.4653255, -0.3326984, 0.4653255, -0.7980239, 0.7980239)
4: (-0.4972759, 0.5038655, -0.4972759, 0.5038655, -1.0011414, 1.0011414)
5: (-0.5259193, 0.6335245, -0.5259193, 0.6335245, -1.1594439, 1.1594439)
6: (-0.3900203, 0.5112956, -0.3900203, 0.5112956, -0.9013159, 0.9013159)
7: (-0.4749381, 0.5195071, -0.4749381, 0.5195071, -0.9944451, 0.9944451)
8: (-0.5138578, 0.6066080, -0.5138578, 0.6066080, -1.1204658, 1.1204658)
9: (-0.4892738, 0.5884779, -0.4892738, 0.5884779, -1.0777516, 1.0777516)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.87 + 2.43 = 3.30 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.8371422, upper bound: 0.8371422
