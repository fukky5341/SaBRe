## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00162


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0036630, 0.0036630)
1: (-0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0009127, 0.0009127)
2: (-0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0048369, 0.0048369)
3: (0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0022016, 0.0022016)
4: (-0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009362, 0.0009362)
5: (-0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0060836, 0.0060836)
6: (0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0015441, 0.0015441)
7: (0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0039950, 0.0039950)
8: (0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0021009, 0.0021009)
9: (-0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0024361, 0.0024361)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.18 + 2.02 = 3.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0020250, upper bound: 0.0020250

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019955, upper bound: 0.0020161
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0020160, upper bound: 0.0019955
time: 1.18 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.48 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.48
Output dim: 0, lower bound: -0.0019955, upper bound: 0.0020161
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.48
Output dim: 0, lower bound: -0.0020160, upper bound: 0.0019955

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0035793, 0.0035995
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008919, 0.0008969
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0047532, 0.0047264
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0021513, 0.0021634
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009200, 0.0009148
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0059782, 0.0059446
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0015088, 0.0015173
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0039037, 0.0039258
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0020529, 0.0020645
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0023939, 0.0023805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019787, upper bound: 0.0020078
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019871, upper bound: 0.0019974
time: 1.15 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0035995, 0.0035793
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008969, 0.0008919
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0047264, 0.0047532
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0021634, 0.0021513
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009148, 0.0009200
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0059446, 0.0059782
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0015173, 0.0015088
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0039258, 0.0039037
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0020645, 0.0020529
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0023805, 0.0023939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019974, upper bound: 0.0019871
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0020077, upper bound: 0.0019787
time: 1.10 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.42 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 0, lower bound: -0.0019787, upper bound: 0.0020078
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 0, lower bound: -0.0019871, upper bound: 0.0019974
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 0, lower bound: -0.0019974, upper bound: 0.0019871
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 0, lower bound: -0.0020077, upper bound: 0.0019787

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0030247, 0.0030287
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007537, 0.0007547
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0039994, 0.0039941
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0018179, 0.0018203
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0007741, 0.0007731
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0050302, 0.0050235
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0012750, 0.0012767
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0032989, 0.0033032
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0017349, 0.0017371
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0020143, 0.0020116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015892, upper bound: 0.0016160
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015892, upper bound: 0.0016160
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0030059, 0.0030450
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007490, 0.0007587
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0040209, 0.0039692
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0018066, 0.0018301
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0007782, 0.0007682
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0050572, 0.0049922
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0012671, 0.0012836
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0032783, 0.0033210
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0017240, 0.0017465
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0020251, 0.0019991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0016019, upper bound: 0.0016049
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0016019, upper bound: 0.0016049
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0030450, 0.0030059
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007587, 0.0007490
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0039692, 0.0040209
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0018301, 0.0018066
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0007682, 0.0007782
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0049922, 0.0050572
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0012836, 0.0012671
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0033210, 0.0032783
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0017465, 0.0017240
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0019991, 0.0020251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0016049, upper bound: 0.0016019
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0016049, upper bound: 0.0016019
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0030287, 0.0030247
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007547, 0.0007537
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0039941, 0.0039994
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0018203, 0.0018179
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0007731, 0.0007741
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0050235, 0.0050302
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0012767, 0.0012750
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0033032, 0.0032989
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0017371, 0.0017349
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0020116, 0.0020143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0016160, upper bound: 0.0015892
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0016160, upper bound: 0.0015892
time: 1.01 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.23 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.0015892, upper bound: 0.0016160
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.0015892, upper bound: 0.0016160
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.0016019, upper bound: 0.0016049
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.0016019, upper bound: 0.0016049
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.0016049, upper bound: 0.0016019
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.0016049, upper bound: 0.0016019
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.0016160, upper bound: 0.0015892
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.0016160, upper bound: 0.0015892

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.20 + 23.29 = 26.49 seconds
