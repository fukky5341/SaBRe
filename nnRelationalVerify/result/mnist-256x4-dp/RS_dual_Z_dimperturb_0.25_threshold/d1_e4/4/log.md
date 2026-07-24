## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00473928


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0093025, -0.0030970, -0.0093025, -0.0030970, -0.0037692, 0.0037692)
1: (-0.0050240, -0.0043312, -0.0050240, -0.0043312, -0.0006928, 0.0006928)
2: (0.0326720, 0.0486743, 0.0326720, 0.0486743, -0.0106962, 0.0106962)
3: (0.0006537, 0.0106900, 0.0006537, 0.0106900, -0.0047478, 0.0047478)
4: (-0.0040766, -0.0022472, -0.0040766, -0.0022472, -0.0018294, 0.0018294)
5: (0.0100550, 0.0117115, 0.0100550, 0.0117115, -0.0016565, 0.0016565)
6: (-0.0165948, -0.0020640, -0.0165948, -0.0020640, -0.0071570, 0.0071570)
7: (0.9555652, 0.9750761, 0.9555652, 0.9750761, -0.0195109, 0.0195109)
8: (-0.0055530, 0.0003554, -0.0055530, 0.0003554, -0.0059084, 0.0059084)
9: (-0.0035633, -0.0009848, -0.0035633, -0.0009848, -0.0025785, 0.0025785)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.58 + 1.32 = 2.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0058820, upper bound: 0.0058820

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0052347, upper bound: 0.0051600
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0051600, upper bound: 0.0052347
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.14
Output dim: 7, lower bound: -0.0052347, upper bound: 0.0051600
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.14
Output dim: 7, lower bound: -0.0051600, upper bound: 0.0052347

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0093025, -0.0030970, -0.0093025, -0.0030970, -0.0037666, 0.0037578
1: -0.0050240, -0.0043312, -0.0050240, -0.0043312, -0.0006928, 0.0006928
2: 0.0326720, 0.0486743, 0.0326720, 0.0486743, -0.0106458, 0.0106997
3: 0.0006537, 0.0106900, 0.0006537, 0.0106900, -0.0047161, 0.0047458
4: -0.0040766, -0.0022472, -0.0040766, -0.0022472, -0.0018294, 0.0018294
5: 0.0100550, 0.0117115, 0.0100550, 0.0117115, -0.0016565, 0.0016565
6: -0.0165948, -0.0020640, -0.0165948, -0.0020640, -0.0071480, 0.0071208
7: 0.9555652, 0.9750761, 0.9555652, 0.9750761, -0.0195109, 0.0195109
8: -0.0055530, 0.0003554, -0.0055530, 0.0003554, -0.0059084, 0.0059084
9: -0.0035633, -0.0009848, -0.0035633, -0.0009848, -0.0025785, 0.0025785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0037585, upper bound: 0.0037585
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0037585, upper bound: 0.0037585
time: 0.45 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0093025, -0.0030970, -0.0093025, -0.0030970, -0.0037578, 0.0037692
1: -0.0050240, -0.0043312, -0.0050240, -0.0043312, -0.0006928, 0.0006928
2: 0.0326720, 0.0486743, 0.0326720, 0.0486743, -0.0106962, 0.0106458
3: 0.0006537, 0.0106900, 0.0006537, 0.0106900, -0.0047478, 0.0047161
4: -0.0040766, -0.0022472, -0.0040766, -0.0022472, -0.0018294, 0.0018294
5: 0.0100550, 0.0117115, 0.0100550, 0.0117115, -0.0016565, 0.0016565
6: -0.0165948, -0.0020640, -0.0165948, -0.0020640, -0.0071208, 0.0071570
7: 0.9555652, 0.9750761, 0.9555652, 0.9750761, -0.0195109, 0.0195109
8: -0.0055530, 0.0003554, -0.0055530, 0.0003554, -0.0059084, 0.0059084
9: -0.0035633, -0.0009848, -0.0035633, -0.0009848, -0.0025785, 0.0025785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0037585, upper bound: 0.0037585
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0037585, upper bound: 0.0037585
time: 0.45 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.38 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.38
Output dim: 7, lower bound: -0.0037585, upper bound: 0.0037585
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.38
Output dim: 7, lower bound: -0.0037585, upper bound: 0.0037585
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.38
Output dim: 7, lower bound: -0.0037585, upper bound: 0.0037585
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.38
Output dim: 7, lower bound: -0.0037585, upper bound: 0.0037585

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.90 + 5.88 = 8.78 seconds
