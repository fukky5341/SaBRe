## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00021588


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0035612, 0.0041214, 0.0035612, 0.0041214, -0.0002843, 0.0002843)
1: (0.0018368, 0.0019177, 0.0018368, 0.0019177, -0.0000411, 0.0000411)
2: (0.0120813, 0.0123910, 0.0120813, 0.0123910, -0.0001572, 0.0001572)
3: (-0.0021854, -0.0018651, -0.0021854, -0.0018651, -0.0001626, 0.0001626)
4: (-0.0020179, -0.0016711, -0.0020179, -0.0016711, -0.0001760, 0.0001760)
5: (0.0056948, 0.0060229, 0.0056948, 0.0060229, -0.0001665, 0.0001665)
6: (0.0002948, 0.0015969, 0.0002948, 0.0015969, -0.0006608, 0.0006608)
7: (-0.0047315, -0.0029583, -0.0047315, -0.0029583, -0.0008999, 0.0008999)
8: (0.9858809, 0.9871300, 0.9858809, 0.9871300, -0.0006339, 0.0006339)
9: (-0.0042048, -0.0030709, -0.0042048, -0.0030709, -0.0005754, 0.0005754)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.48 + 1.21 = 2.69 seconds
status: Status.ADV_EXAMPLE
