## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.22685915495631442


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.3077660, -4.8719282, -5.3077660, -4.8719282, -0.2692952, 0.2692952)
1: (-8.8621044, -8.2962799, -8.8621044, -8.2962799, -0.3403883, 0.3403883)
2: (-0.3782907, 0.1426924, -0.3782907, 0.1426924, -0.3090377, 0.3090382)
3: (3.6921375, 4.2909002, 3.6921375, 4.2909002, -0.3293557, 0.3293560)
4: (-14.9872875, -14.2317591, -14.9872875, -14.2317591, -0.3360329, 0.3360329)
5: (-6.8909402, -6.4223480, -6.8909402, -6.4223480, -0.2533360, 0.2533360)
6: (-6.5853829, -5.9237232, -6.5853829, -5.9237232, -0.3130364, 0.3130364)
7: (-9.1183453, -8.5025358, -9.1183453, -8.5025358, -0.2528201, 0.2528201)
8: (-4.9428225, -4.5061908, -4.9428225, -4.5061908, -0.2149245, 0.2149246)
9: (-11.9950161, -11.4004326, -11.9950161, -11.4004326, -0.3242459, 0.3242459)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 20.61 + 35.48 = 56.09 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -0.1890623, upper bound: 0.1890623
