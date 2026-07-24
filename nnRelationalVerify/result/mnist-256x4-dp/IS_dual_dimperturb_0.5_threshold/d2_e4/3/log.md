## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.40684923


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0857445, 0.1024349, -0.0857445, 0.1024349, -0.1881793, 0.1881793)
1: (0.7811663, 1.0518622, 0.7811663, 1.0518622, -0.2706960, 0.2706960)
2: (-0.0651192, 0.1121050, -0.0651192, 0.1121050, -0.1772242, 0.1772242)
3: (-0.0436161, 0.0819061, -0.0436161, 0.0819061, -0.1255222, 0.1255222)
4: (-0.0775046, 0.0515187, -0.0775046, 0.0515187, -0.1290233, 0.1290233)
5: (-0.0598836, 0.0756878, -0.0598836, 0.0756878, -0.1355714, 0.1355714)
6: (-0.0984564, 0.0788304, -0.0984564, 0.0788304, -0.1772868, 0.1772868)
7: (-0.0760526, 0.1116772, -0.0760526, 0.1116772, -0.1877298, 0.1877298)
8: (-0.0455291, 0.1160677, -0.0455291, 0.1160677, -0.1615967, 0.1615967)
9: (-0.0973068, 0.0911774, -0.0973068, 0.0911774, -0.1884842, 0.1884842)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 2.06 = 3.56 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.2382122, upper bound: 0.2382122
