## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00079287


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0017176, -0.0000457, -0.0017176, -0.0000457, -0.0013895, 0.0013895)
1: (-0.0043522, -0.0037928, -0.0043522, -0.0037928, -0.0004844, 0.0004844)
2: (0.0123517, 0.0146074, 0.0123517, 0.0146074, -0.0018223, 0.0018223)
3: (1.0079064, 1.0093433, 1.0079064, 1.0093433, -0.0014368, 0.0014368)
4: (-0.0039872, -0.0036148, -0.0039872, -0.0036148, -0.0002928, 0.0002928)
5: (0.0026328, 0.0039221, 0.0026328, 0.0039221, -0.0010671, 0.0010671)
6: (-0.0024909, -0.0023560, -0.0024909, -0.0023560, -0.0001349, 0.0001349)
7: (-0.0129905, -0.0108531, -0.0129905, -0.0108531, -0.0021052, 0.0021052)
8: (-0.0106224, -0.0065648, -0.0106224, -0.0065648, -0.0031411, 0.0031411)
9: (-0.0010504, 0.0009726, -0.0010504, 0.0009726, -0.0015421, 0.0015421)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.53 + 1.54 = 3.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0008353, upper bound: 0.0008353

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008072, upper bound: 0.0008070
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008070, upper bound: 0.0008072
time: 0.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 3, lower bound: -0.0008072, upper bound: 0.0008070
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 3, lower bound: -0.0008070, upper bound: 0.0008072

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017176, -0.0000457, -0.0017176, -0.0000457, -0.0013826, 0.0013855
1: -0.0043522, -0.0037928, -0.0043522, -0.0037928, -0.0004811, 0.0004825
2: 0.0123517, 0.0146074, 0.0123517, 0.0146074, -0.0018120, 0.0018165
3: 1.0079064, 1.0093433, 1.0079064, 1.0093433, -0.0014368, 0.0014368
4: -0.0039872, -0.0036148, -0.0039872, -0.0036148, -0.0002916, 0.0002908
5: 0.0026328, 0.0039221, 0.0026328, 0.0039221, -0.0010617, 0.0010640
6: -0.0024909, -0.0023560, -0.0024909, -0.0023560, -0.0001349, 0.0001349
7: -0.0129905, -0.0108531, -0.0129905, -0.0108531, -0.0021048, 0.0021045
8: -0.0106224, -0.0065648, -0.0106224, -0.0065648, -0.0031269, 0.0031171
9: -0.0010504, 0.0009726, -0.0010504, 0.0009726, -0.0015301, 0.0015353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007869, upper bound: 0.0007872
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007872, upper bound: 0.0007869
time: 0.59 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017176, -0.0000457, -0.0017176, -0.0000457, -0.0013895, 0.0013826
1: -0.0043522, -0.0037928, -0.0043522, -0.0037928, -0.0004844, 0.0004811
2: 0.0123517, 0.0146074, 0.0123517, 0.0146074, -0.0018223, 0.0018120
3: 1.0079064, 1.0093433, 1.0079064, 1.0093433, -0.0014368, 0.0014368
4: -0.0039872, -0.0036148, -0.0039872, -0.0036148, -0.0002908, 0.0002928
5: 0.0026328, 0.0039221, 0.0026328, 0.0039221, -0.0010671, 0.0010617
6: -0.0024909, -0.0023560, -0.0024909, -0.0023560, -0.0001349, 0.0001349
7: -0.0129905, -0.0108531, -0.0129905, -0.0108531, -0.0021045, 0.0021052
8: -0.0106224, -0.0065648, -0.0106224, -0.0065648, -0.0031171, 0.0031411
9: -0.0010504, 0.0009726, -0.0010504, 0.0009726, -0.0015421, 0.0015301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007869, upper bound: 0.0007872
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007872, upper bound: 0.0007869
time: 0.58 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.74
Output dim: 3, lower bound: -0.0007869, upper bound: 0.0007872
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.74
Output dim: 3, lower bound: -0.0007872, upper bound: 0.0007869
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.74
Output dim: 3, lower bound: -0.0007869, upper bound: 0.0007872
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.74
Output dim: 3, lower bound: -0.0007872, upper bound: 0.0007869

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.07 + 7.09 = 10.16 seconds
