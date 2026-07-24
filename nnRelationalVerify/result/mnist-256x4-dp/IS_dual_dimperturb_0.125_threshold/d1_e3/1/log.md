## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00010188


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040959, -0.0040893, -0.0040959, -0.0040893, -0.0000030, 0.0000030)
1: (-0.0060482, -0.0057986, -0.0060482, -0.0057986, -0.0001135, 0.0001135)
2: (0.9692054, 0.9695050, 0.9692054, 0.9695050, -0.0001362, 0.0001362)
3: (0.0191697, 0.0213791, 0.0191697, 0.0213791, -0.0010046, 0.0010046)
4: (-0.0023190, -0.0021510, -0.0023190, -0.0021510, -0.0000764, 0.0000764)
5: (0.0149265, 0.0150964, 0.0149265, 0.0150964, -0.0000772, 0.0000772)
6: (0.0045639, 0.0046465, 0.0045639, 0.0046465, -0.0000376, 0.0000376)
7: (-0.0133188, -0.0127462, -0.0133188, -0.0127462, -0.0002603, 0.0002603)
8: (0.0061626, 0.0066169, 0.0061626, 0.0066169, -0.0002065, 0.0002065)
9: (0.0088087, 0.0096257, 0.0088087, 0.0096257, -0.0003715, 0.0003715)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.25 = 2.57 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0000901, upper bound: 0.0000901
