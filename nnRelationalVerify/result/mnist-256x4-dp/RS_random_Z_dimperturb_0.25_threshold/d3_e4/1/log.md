## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00167283


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0013079, 0.0000088, -0.0013079, 0.0000088, -0.0008808, 0.0008808)
1: (-0.0076295, -0.0042882, -0.0076295, -0.0042882, -0.0022352, 0.0022352)
2: (0.0302967, 0.0323696, 0.0302967, 0.0323696, -0.0013867, 0.0013867)
3: (-0.0008798, 0.0029910, -0.0008798, 0.0029910, -0.0025894, 0.0025894)
4: (-0.0066535, -0.0032548, -0.0066535, -0.0032548, -0.0022736, 0.0022736)
5: (0.0112180, 0.0125053, 0.0112180, 0.0125053, -0.0008612, 0.0008612)
6: (-0.0007284, 0.0041841, -0.0007284, 0.0041841, -0.0032862, 0.0032862)
7: (0.9775496, 0.9809871, 0.9775496, 0.9809871, -0.0022995, 0.0022995)
8: (-0.0106346, -0.0069491, -0.0106346, -0.0069491, -0.0024655, 0.0024655)
9: (-0.0004093, 0.0020252, -0.0004093, 0.0020252, -0.0016286, 0.0016286)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.86 + 1.74 = 3.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0016822, upper bound: 0.0016822

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016592, upper bound: 0.0016316
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016316, upper bound: 0.0016592
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.51 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.51
Output dim: 7, lower bound: -0.0016592, upper bound: 0.0016316
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.51
Output dim: 7, lower bound: -0.0016316, upper bound: 0.0016592

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.60 + 1.51 = 5.12 seconds
