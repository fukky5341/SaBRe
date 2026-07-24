## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00046665


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0026198, -0.0021603, -0.0026198, -0.0021603, -0.0002412, 0.0002412)
1: (-0.0109587, -0.0097926, -0.0109587, -0.0097926, -0.0006121, 0.0006121)
2: (0.0282312, 0.0289546, 0.0282312, 0.0289546, -0.0003798, 0.0003798)
3: (0.0054969, 0.0068477, 0.0054969, 0.0068477, -0.0007091, 0.0007091)
4: (-0.0100399, -0.0088538, -0.0100399, -0.0088538, -0.0006226, 0.0006226)
5: (0.0099353, 0.0103846, 0.0099353, 0.0103846, -0.0002358, 0.0002358)
6: (0.0073644, 0.0090788, 0.0073644, 0.0090788, -0.0009000, 0.0009000)
7: (0.9832125, 0.9844122, 0.9832125, 0.9844122, -0.0006298, 0.0006298)
8: (-0.0045631, -0.0032769, -0.0045631, -0.0032769, -0.0006752, 0.0006752)
9: (-0.0028351, -0.0019854, -0.0028351, -0.0019854, -0.0004460, 0.0004460)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.73 + 1.34 = 3.07 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0004405, upper bound: 0.0004406
