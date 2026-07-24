## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.603988435


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.4247053, 1.0629222, 0.4247053, 1.0629222, -0.6382170, 0.6382170)
1: (-0.2708166, 0.2646017, -0.2708166, 0.2646017, -0.5354183, 0.5354183)
2: (-0.1877118, 0.3586684, -0.1877118, 0.3586684, -0.5463802, 0.5463802)
3: (-0.1972049, 0.2679627, -0.1972049, 0.2679627, -0.4651676, 0.4651676)
4: (-0.2848693, 0.2447191, -0.2848693, 0.2447191, -0.5295883, 0.5295883)
5: (-0.3054801, 0.4001342, -0.3054801, 0.4001342, -0.7056143, 0.7056143)
6: (-0.2139553, 0.2947329, -0.2139553, 0.2947329, -0.5086882, 0.5086882)
7: (-0.2969876, 0.3016686, -0.2969876, 0.3016686, -0.5986562, 0.5986562)
8: (-0.2833509, 0.3674826, -0.2833509, 0.3674826, -0.6508335, 0.6508335)
9: (-0.2752340, 0.3545616, -0.2752340, 0.3545616, -0.6297956, 0.6297956)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.77 + 2.63 = 4.40 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.5270190, upper bound: 0.5270190
