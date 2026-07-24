## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0018212


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0002407, -0.0000688, -0.0002407, -0.0000688, -0.0001217, 0.0001217)
1: (0.0001933, 0.0009980, 0.0001933, 0.0009980, -0.0005697, 0.0005697)
2: (0.0148453, 0.0160505, 0.0148453, 0.0160505, -0.0008532, 0.0008532)
3: (0.0005361, 0.0014424, 0.0005361, 0.0014424, -0.0006416, 0.0006416)
4: (-0.0038851, -0.0030492, -0.0038851, -0.0030492, -0.0005918, 0.0005918)
5: (0.0084733, 0.0093779, 0.0084733, 0.0093779, -0.0006404, 0.0006404)
6: (0.0093939, 0.0097353, 0.0093939, 0.0097353, -0.0002417, 0.0002417)
7: (-0.0187579, -0.0167941, -0.0187579, -0.0167941, -0.0013903, 0.0013903)
8: (0.9700473, 0.9756737, 0.9700473, 0.9756737, -0.0039833, 0.0039833)
9: (0.0048026, 0.0064563, 0.0048026, 0.0064563, -0.0011707, 0.0011707)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.19 = 2.47 seconds
status: Status.ADV_EXAMPLE
