## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.04893569308


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0207791, 0.0038929, -0.0207791, 0.0038929, -0.0246720, 0.0246720)
1: (-0.0107140, 0.0097190, -0.0107140, 0.0097190, -0.0204330, 0.0204330)
2: (0.0013663, 0.0301719, 0.0013663, 0.0301719, -0.0288057, 0.0288057)
3: (-0.0095223, 0.0109147, -0.0095223, 0.0109147, -0.0204370, 0.0204370)
4: (-0.0186100, 0.0155992, -0.0186100, 0.0155992, -0.0342093, 0.0342093)
5: (-0.0119216, 0.0291928, -0.0119216, 0.0291928, -0.0411144, 0.0411144)
6: (-0.0050545, 0.0194704, -0.0050545, 0.0194704, -0.0245250, 0.0245250)
7: (-0.0332284, 0.0028428, -0.0332284, 0.0028428, -0.0360712, 0.0360712)
8: (0.9371550, 0.9884812, 0.9371550, 0.9884812, -0.0513262, 0.0513262)
9: (-0.0018395, 0.0441409, -0.0018395, 0.0441409, -0.0459805, 0.0459805)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 2.41 = 3.79 seconds
status: Status.VERIFIED
relational distance
Output dim: 8, lower bound: -0.0437944, upper bound: 0.0437944
