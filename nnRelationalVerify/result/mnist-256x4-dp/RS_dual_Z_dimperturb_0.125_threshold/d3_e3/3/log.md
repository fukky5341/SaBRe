## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00010616


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0126821, -0.0118625, -0.0126821, -0.0118625, -0.0005067, 0.0005067)
1: (-0.0065142, -0.0062832, -0.0065142, -0.0062832, -0.0001429, 0.0001429)
2: (-0.0095035, -0.0077987, -0.0095035, -0.0077987, -0.0010541, 0.0010541)
3: (0.0003697, 0.0005953, 0.0003697, 0.0005953, -0.0001395, 0.0001395)
4: (0.0119201, 0.0131942, 0.0119201, 0.0131942, -0.0007878, 0.0007878)
5: (0.9988180, 0.9991720, 0.9988180, 0.9991720, -0.0002189, 0.0002189)
6: (0.0068107, 0.0071321, 0.0068107, 0.0071321, -0.0001987, 0.0001987)
7: (0.0020350, 0.0032341, 0.0020350, 0.0032341, -0.0007414, 0.0007414)
8: (-0.0117099, -0.0107767, -0.0117099, -0.0107767, -0.0005770, 0.0005770)
9: (-0.0030800, -0.0029995, -0.0030800, -0.0029995, -0.0000498, 0.0000498)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.82 + 1.22 = 3.04 seconds
status: Status.ADV_EXAMPLE
