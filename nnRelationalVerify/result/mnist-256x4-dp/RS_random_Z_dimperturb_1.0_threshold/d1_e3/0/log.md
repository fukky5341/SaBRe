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
execution time: IAR + RelationalAnalysis = 1.23 + 2.03 = 3.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0020250, upper bound: 0.0020250

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019309, upper bound: 0.0019391
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019391, upper bound: 0.0019310
time: 1.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.72
Output dim: 0, lower bound: -0.0019309, upper bound: 0.0019391
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.72
Output dim: 0, lower bound: -0.0019391, upper bound: 0.0019310

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0036315, 0.0036425
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0009049, 0.0009076
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0048099, 0.0047953
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0021826, 0.0021892
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009309, 0.0009281
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0060495, 0.0060312
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0015308, 0.0015354
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0039606, 0.0039726
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0020828, 0.0020892
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0024225, 0.0024152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019061, upper bound: 0.0019304
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019221, upper bound: 0.0019116
time: 1.23 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0036630, 0.0036315
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0009127, 0.0009049
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0047953, 0.0048369
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0022016, 0.0021826
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009281, 0.0009362
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0060312, 0.0060836
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0015441, 0.0015308
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0039950, 0.0039606
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0021009, 0.0020828
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0024152, 0.0024361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019207, upper bound: 0.0019225
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019308, upper bound: 0.0019147
time: 1.24 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.54 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 0, lower bound: -0.0019061, upper bound: 0.0019304
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 0, lower bound: -0.0019221, upper bound: 0.0019116
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 0, lower bound: -0.0019207, upper bound: 0.0019225
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 0, lower bound: -0.0019308, upper bound: 0.0019147

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0035473, 0.0035798
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008839, 0.0008920
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0047271, 0.0046842
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0021320, 0.0021516
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009149, 0.0009066
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0059454, 0.0058914
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0014953, 0.0015090
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0038688, 0.0039043
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0020346, 0.0020532
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0023808, 0.0023592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018161, upper bound: 0.0018590
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018367, upper bound: 0.0018416
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0035675, 0.0035583
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008889, 0.0008866
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0046987, 0.0047109
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0021442, 0.0021387
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009094, 0.0009118
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0059098, 0.0059251
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0015038, 0.0015000
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0038909, 0.0038809
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0020462, 0.0020409
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0023665, 0.0023727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018860, upper bound: 0.0018566
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018621, upper bound: 0.0018719
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0031652, 0.0031150
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007887, 0.0007762
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0041134, 0.0041796
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0019024, 0.0018722
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0007961, 0.0008090
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0051735, 0.0052568
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0013342, 0.0013131
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0034521, 0.0033974
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0018154, 0.0017866
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0020717, 0.0021051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018085, upper bound: 0.0016932
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017014, upper bound: 0.0018128
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0031463, 0.0031339
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007840, 0.0007809
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0041382, 0.0041547
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0018910, 0.0018836
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008009, 0.0008041
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0052048, 0.0052255
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0013263, 0.0013210
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0034315, 0.0034179
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0018046, 0.0017975
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0020842, 0.0020925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018191, upper bound: 0.0016834
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017147, upper bound: 0.0018035
time: 1.10 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.30 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -0.0018161, upper bound: 0.0018590
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -0.0018367, upper bound: 0.0018416
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -0.0018860, upper bound: 0.0018566
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -0.0018621, upper bound: 0.0018719
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -0.0018085, upper bound: 0.0016932
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -0.0017014, upper bound: 0.0018128
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -0.0018191, upper bound: 0.0016834
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -0.0017147, upper bound: 0.0018035

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0032513, 0.0033030
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008101, 0.0008230
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0043616, 0.0042933
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0019541, 0.0019852
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008442, 0.0008310
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0054858, 0.0053999
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0013705, 0.0013923
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0035460, 0.0036024
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0018648, 0.0018945
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0021967, 0.0021624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015235, upper bound: 0.0015565
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015235, upper bound: 0.0015565
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0032766, 0.0032838
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008164, 0.0008182
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0043363, 0.0043267
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0019693, 0.0019737
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008393, 0.0008374
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0054539, 0.0054419
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0013812, 0.0013843
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0035736, 0.0035815
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0018793, 0.0018835
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0021840, 0.0021792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018206, upper bound: 0.0018332
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018282, upper bound: 0.0018238
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0035315, 0.0034941
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008799, 0.0008706
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0046140, 0.0046633
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0021225, 0.0021001
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008930, 0.0009026
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0058032, 0.0058652
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0014886, 0.0014729
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0038516, 0.0038109
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0020255, 0.0020041
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0023238, 0.0023487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015920, upper bound: 0.0015608
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015920, upper bound: 0.0015608
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0035034, 0.0035177
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008729, 0.0008765
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0046451, 0.0046262
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0021056, 0.0021142
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008990, 0.0008954
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0058423, 0.0058185
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0014768, 0.0014828
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0038209, 0.0038366
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0020094, 0.0020176
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0023395, 0.0023300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017735, upper bound: 0.0018033
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017892, upper bound: 0.0017814
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0029403, 0.0028002
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007326, 0.0006977
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0036976, 0.0038827
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0017672, 0.0016830
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0007157, 0.0007515
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0046506, 0.0048834
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0012394, 0.0011804
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0032068, 0.0030540
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0016864, 0.0016061
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0018623, 0.0019555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014700, upper bound: 0.0014137
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014700, upper bound: 0.0014137
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0028504, 0.0028916
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007102, 0.0007205
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0038184, 0.0037639
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0017131, 0.0017380
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0007390, 0.0007285
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0048025, 0.0047340
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0012015, 0.0012189
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0031087, 0.0031537
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0016348, 0.0016585
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0019231, 0.0018957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016403, upper bound: 0.0018100
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016988, upper bound: 0.0017501
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0029230, 0.0028190
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007283, 0.0007024
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0037225, 0.0038597
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0017568, 0.0016943
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0007205, 0.0007470
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0046819, 0.0048545
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0012321, 0.0011883
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0031879, 0.0030746
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0016765, 0.0016169
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0018749, 0.0019440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017504, upper bound: 0.0016807
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018164, upper bound: 0.0016368
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0028315, 0.0029090
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007055, 0.0007248
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0038413, 0.0037390
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0017018, 0.0017484
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0007435, 0.0007237
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0048314, 0.0047026
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0011936, 0.0012262
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0030882, 0.0031727
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0016240, 0.0016685
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0019347, 0.0018831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016276, upper bound: 0.0017317
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016414, upper bound: 0.0017137
time: 1.17 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0015235, upper bound: 0.0015565
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0015235, upper bound: 0.0015565
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0018206, upper bound: 0.0018332
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0018282, upper bound: 0.0018238
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0015920, upper bound: 0.0015608
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0015920, upper bound: 0.0015608
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0017735, upper bound: 0.0018033
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0017892, upper bound: 0.0017814
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0014700, upper bound: 0.0014137
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0014700, upper bound: 0.0014137
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0016403, upper bound: 0.0018100
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0016988, upper bound: 0.0017501
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0017504, upper bound: 0.0016807
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0018164, upper bound: 0.0016368
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0016276, upper bound: 0.0017317
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 0, lower bound: -0.0016414, upper bound: 0.0017137

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0026621, 0.0026566
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006633, 0.0006620
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0035080, 0.0035153
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0016000, 0.0015967
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006790, 0.0006804
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0044122, 0.0044213
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0011222, 0.0011199
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0029034, 0.0028974
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0015269, 0.0015237
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017668, 0.0017705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017095, upper bound: 0.0016190
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015809, upper bound: 0.0017198
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0026475, 0.0026693
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006597, 0.0006651
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0035248, 0.0034960
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015912, 0.0016043
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006822, 0.0006766
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0044333, 0.0043970
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0011160, 0.0011252
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0028875, 0.0029113
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0015185, 0.0015310
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017753, 0.0017608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017176, upper bound: 0.0016071
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015941, upper bound: 0.0017109
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0032325, 0.0032717
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008054, 0.0008152
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0043202, 0.0042684
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0019428, 0.0019664
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008362, 0.0008261
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0054337, 0.0053686
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0013626, 0.0013791
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0035255, 0.0035682
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0018540, 0.0018765
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0021759, 0.0021498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017064, upper bound: 0.0018008
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017710, upper bound: 0.0017476
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0032536, 0.0032468
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008107, 0.0008090
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0042874, 0.0042963
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0019555, 0.0019514
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008298, 0.0008315
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0053924, 0.0054037
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0013715, 0.0013686
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0035485, 0.0035411
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0018661, 0.0018622
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0021593, 0.0021639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017243, upper bound: 0.0017788
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017867, upper bound: 0.0017242
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0027507, 0.0028487
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006854, 0.0007098
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0037617, 0.0036323
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0016533, 0.0017122
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0007281, 0.0007030
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0047312, 0.0045685
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0011595, 0.0012008
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0030000, 0.0031069
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0015777, 0.0016339
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0018946, 0.0018294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016045, upper bound: 0.0017455
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015775, upper bound: 0.0017741
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0027951, 0.0027963
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006965, 0.0006968
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0036925, 0.0036910
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0016800, 0.0016807
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0007147, 0.0007144
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0046442, 0.0046423
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0011783, 0.0011787
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0030485, 0.0030498
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0016032, 0.0016038
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0018597, 0.0018590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016625, upper bound: 0.0017409
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016902, upper bound: 0.0017302
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0028233, 0.0027682
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007035, 0.0006898
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0036553, 0.0037281
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0016969, 0.0016637
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0007075, 0.0007216
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0045974, 0.0046890
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0011901, 0.0011669
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0030792, 0.0030191
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0016193, 0.0015877
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0018410, 0.0018777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017154, upper bound: 0.0016206
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016802, upper bound: 0.0016421
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0028757, 0.0027237
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007165, 0.0006787
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0035966, 0.0037973
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0017284, 0.0016370
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006961, 0.0007350
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0045236, 0.0047760
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0012122, 0.0011481
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0031364, 0.0029706
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0016494, 0.0015622
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0018115, 0.0019125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017262, upper bound: 0.0015655
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017448, upper bound: 0.0015504
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025100, 0.0026126
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006254, 0.0006510
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0034499, 0.0033145
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015086, 0.0015703
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006677, 0.0006415
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0043391, 0.0041688
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010581, 0.0011013
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027376, 0.0028494
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014397, 0.0014985
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017376, 0.0016694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015892, upper bound: 0.0017226
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016190, upper bound: 0.0017095
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025396, 0.0025873
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006328, 0.0006447
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0034165, 0.0033535
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015264, 0.0015550
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006613, 0.0006491
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0042970, 0.0042178
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010705, 0.0010906
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027698, 0.0028218
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014566, 0.0014840
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017207, 0.0016890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016099, upper bound: 0.0017046
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016328, upper bound: 0.0016893
time: 1.12 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.35 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0017095, upper bound: 0.0016190
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0015809, upper bound: 0.0017198
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0017176, upper bound: 0.0016071
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0015941, upper bound: 0.0017109
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0017064, upper bound: 0.0018008
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0017710, upper bound: 0.0017476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0017243, upper bound: 0.0017788
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0017867, upper bound: 0.0017242
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0016045, upper bound: 0.0017455
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0015775, upper bound: 0.0017741
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0016625, upper bound: 0.0017409
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0016902, upper bound: 0.0017302
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0017154, upper bound: 0.0016206
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0016802, upper bound: 0.0016421
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0017262, upper bound: 0.0015655
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0017448, upper bound: 0.0015504
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0015892, upper bound: 0.0017226
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0016190, upper bound: 0.0017095
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0016099, upper bound: 0.0017046
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.0016328, upper bound: 0.0016893

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0023872, 0.0022917
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005948, 0.0005710
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0030262, 0.0031522
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014348, 0.0013774
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005857, 0.0006101
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0038061, 0.0039647
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010063, 0.0009660
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0026035, 0.0024994
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013692, 0.0013144
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015241, 0.0015876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016557, upper bound: 0.0016164
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017068, upper bound: 0.0015558
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022972, 0.0023722
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005724, 0.0005911
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031325, 0.0030334
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013807, 0.0014258
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006063, 0.0005871
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0039398, 0.0038152
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009683, 0.0010000
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0025054, 0.0025872
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013176, 0.0013606
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015777, 0.0015278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015390, upper bound: 0.0017172
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015781, upper bound: 0.0016537
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0023695, 0.0023044
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005904, 0.0005742
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0030429, 0.0031289
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014241, 0.0013850
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005890, 0.0006056
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0038272, 0.0039353
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009988, 0.0009714
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0025842, 0.0025133
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013590, 0.0013217
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015326, 0.0015759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016816, upper bound: 0.0015493
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016529, upper bound: 0.0015675
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022826, 0.0023870
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005688, 0.0005948
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031520, 0.0030141
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013719, 0.0014346
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006101, 0.0005834
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0039644, 0.0037910
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009622, 0.0010062
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024895, 0.0026033
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013092, 0.0013691
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015875, 0.0015181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013094, upper bound: 0.0013721
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013094, upper bound: 0.0013721
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0031698, 0.0032472
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007898, 0.0008091
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0042879, 0.0041857
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0019051, 0.0019517
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008299, 0.0008101
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0053931, 0.0052645
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0013362, 0.0013688
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0034571, 0.0035416
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0018181, 0.0018625
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0021596, 0.0021081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015894, upper bound: 0.0015705
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015004, upper bound: 0.0016900
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0032180, 0.0032090
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008018, 0.0007996
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0042375, 0.0042493
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0019341, 0.0019287
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008202, 0.0008224
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0053297, 0.0053445
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0013565, 0.0013527
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0035096, 0.0034999
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0018457, 0.0018406
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0021342, 0.0021402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016559, upper bound: 0.0015236
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015468, upper bound: 0.0016348
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0031909, 0.0032240
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007951, 0.0008033
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0042572, 0.0042136
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0019178, 0.0019377
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008240, 0.0008155
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0053545, 0.0052996
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0013451, 0.0013590
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0034802, 0.0035162
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0018302, 0.0018491
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0021442, 0.0021222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017008, upper bound: 0.0017703
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017159, upper bound: 0.0017627
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0032343, 0.0031841
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008059, 0.0007934
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0042046, 0.0042709
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0019439, 0.0019138
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008138, 0.0008266
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0052883, 0.0053717
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0013634, 0.0013422
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0035275, 0.0034728
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0018551, 0.0018263
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0021177, 0.0021511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017633, upper bound: 0.0017157
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017783, upper bound: 0.0017104
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025800, 0.0026483
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006429, 0.0006599
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0034970, 0.0034068
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015506, 0.0015917
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006768, 0.0006594
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0043984, 0.0042849
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010876, 0.0011163
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0028138, 0.0028883
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014798, 0.0015189
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017613, 0.0017159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015746, upper bound: 0.0017363
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015957, upper bound: 0.0017241
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025537, 0.0026665
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006363, 0.0006644
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0035211, 0.0033721
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015348, 0.0016027
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006815, 0.0006527
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0044286, 0.0042412
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010765, 0.0011240
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027851, 0.0029082
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014647, 0.0015294
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017734, 0.0016984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013187, upper bound: 0.0014381
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013187, upper bound: 0.0014381
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0027010, 0.0027206
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006730, 0.0006779
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0035926, 0.0035666
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0016234, 0.0016352
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006953, 0.0006903
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0045185, 0.0044859
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0011386, 0.0011468
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0029458, 0.0029672
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0015492, 0.0015604
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0018094, 0.0017963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013700, upper bound: 0.0014299
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013700, upper bound: 0.0014299
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0027250, 0.0027031
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006790, 0.0006735
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0035694, 0.0035983
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0016378, 0.0016246
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006908, 0.0006964
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0044893, 0.0045257
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0011487, 0.0011394
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0029720, 0.0029481
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0015629, 0.0015504
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017977, 0.0018123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013955, upper bound: 0.0014220
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013955, upper bound: 0.0014220
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0026507, 0.0025677
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006605, 0.0006398
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0033907, 0.0035002
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015931, 0.0015433
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006563, 0.0006775
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0042646, 0.0044023
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0011174, 0.0010824
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0028910, 0.0028005
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0015203, 0.0014727
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017077, 0.0017629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016952, upper bound: 0.0016119
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017063, upper bound: 0.0015924
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0026263, 0.0025882
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006544, 0.0006449
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0034177, 0.0034680
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015785, 0.0015556
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006615, 0.0006712
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0042986, 0.0043618
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0011071, 0.0010910
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0028643, 0.0028228
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0015063, 0.0014845
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017213, 0.0017466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016631, upper bound: 0.0016334
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016708, upper bound: 0.0016091
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025623, 0.0024348
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006384, 0.0006067
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0032151, 0.0033834
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015400, 0.0014634
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006223, 0.0006549
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0040438, 0.0042555
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010801, 0.0010263
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027945, 0.0026555
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014696, 0.0013965
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0016193, 0.0017041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017004, upper bound: 0.0015568
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017172, upper bound: 0.0015390
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025841, 0.0024095
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006439, 0.0006004
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031817, 0.0034123
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015531, 0.0014482
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006158, 0.0006604
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0040018, 0.0042917
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010893, 0.0010157
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0028183, 0.0026279
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014821, 0.0013820
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0016025, 0.0017186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017076, upper bound: 0.0014865
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016827, upper bound: 0.0015154
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022843, 0.0024023
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005692, 0.0005986
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031722, 0.0030164
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013729, 0.0014439
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006140, 0.0005838
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0039898, 0.0037938
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009629, 0.0010127
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024913, 0.0026200
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013102, 0.0013779
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015977, 0.0015192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012923, upper bound: 0.0013887
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012923, upper bound: 0.0013887
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0023071, 0.0023872
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005749, 0.0005948
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031522, 0.0030465
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013867, 0.0014348
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006101, 0.0005897
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0039647, 0.0038318
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009725, 0.0010063
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0025163, 0.0026035
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013233, 0.0013692
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015876, 0.0015344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015769, upper bound: 0.0016395
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015667, upper bound: 0.0016739
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0023138, 0.0023783
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005765, 0.0005926
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031405, 0.0030554
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013907, 0.0014294
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006078, 0.0005914
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0039499, 0.0038429
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009754, 0.0010025
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0025236, 0.0025938
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013271, 0.0013641
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015817, 0.0015389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013094, upper bound: 0.0013721
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013094, upper bound: 0.0013720
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0023315, 0.0023619
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005809, 0.0005885
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031188, 0.0030787
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014013, 0.0014195
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006036, 0.0005959
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0039226, 0.0038722
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009828, 0.0009956
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0025428, 0.0025759
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013372, 0.0013547
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015708, 0.0015506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015710, upper bound: 0.0016866
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016301, upper bound: 0.0016364
time: 1.13 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.40 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0016557, upper bound: 0.0016164
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0017068, upper bound: 0.0015558
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0015390, upper bound: 0.0017172
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0015781, upper bound: 0.0016537
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0016816, upper bound: 0.0015493
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0016529, upper bound: 0.0015675
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0013094, upper bound: 0.0013721
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0013094, upper bound: 0.0013721
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0015894, upper bound: 0.0015705
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0015004, upper bound: 0.0016900
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0016559, upper bound: 0.0015236
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0015468, upper bound: 0.0016348
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0017008, upper bound: 0.0017703
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0017159, upper bound: 0.0017627
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0017633, upper bound: 0.0017157
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0017783, upper bound: 0.0017104
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0015746, upper bound: 0.0017363
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0015957, upper bound: 0.0017241
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0013187, upper bound: 0.0014381
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0013187, upper bound: 0.0014381
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0013700, upper bound: 0.0014299
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0013700, upper bound: 0.0014299
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0013955, upper bound: 0.0014220
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0013955, upper bound: 0.0014220
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0016952, upper bound: 0.0016119
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0017063, upper bound: 0.0015924
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0016631, upper bound: 0.0016334
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0016708, upper bound: 0.0016091
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0017004, upper bound: 0.0015568
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0017172, upper bound: 0.0015390
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0017076, upper bound: 0.0014865
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0016827, upper bound: 0.0015154
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0012923, upper bound: 0.0013887
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0012923, upper bound: 0.0013887
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0015769, upper bound: 0.0016395
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0015667, upper bound: 0.0016739
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0013094, upper bound: 0.0013721
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0013094, upper bound: 0.0013720
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0015710, upper bound: 0.0016866
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0016301, upper bound: 0.0016364

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0023475, 0.0023021
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005849, 0.0005736
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0030400, 0.0030999
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014109, 0.0013837
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005884, 0.0006000
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0038235, 0.0038989
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009896, 0.0009704
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0025603, 0.0025108
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013465, 0.0013204
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015311, 0.0015613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013412, upper bound: 0.0013139
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013412, upper bound: 0.0013139
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0023927, 0.0022521
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005962, 0.0005612
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0029739, 0.0031596
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014381, 0.0013536
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005756, 0.0006115
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0037403, 0.0039739
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010086, 0.0009493
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0026096, 0.0024562
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013724, 0.0012917
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014978, 0.0015913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013704, upper bound: 0.0012865
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013704, upper bound: 0.0012865
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022576, 0.0023834
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005625, 0.0005939
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031472, 0.0029811
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013569, 0.0014325
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006091, 0.0005770
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0039584, 0.0037494
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009516, 0.0010047
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024622, 0.0025994
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012948, 0.0013670
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015851, 0.0015014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012768, upper bound: 0.0013804
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012768, upper bound: 0.0013804
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022969, 0.0023326
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005723, 0.0005812
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0030801, 0.0030330
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013805, 0.0014019
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005962, 0.0005870
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0038740, 0.0038148
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009682, 0.0009833
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0025051, 0.0025440
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013174, 0.0013379
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015513, 0.0015276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012986, upper bound: 0.0013437
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012986, upper bound: 0.0013437
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022557, 0.0021597
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005621, 0.0005381
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0028519, 0.0029786
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013557, 0.0012980
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005520, 0.0005765
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0035869, 0.0037463
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009508, 0.0009104
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024601, 0.0023555
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012938, 0.0012387
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014364, 0.0015002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016265, upper bound: 0.0015467
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016790, upper bound: 0.0014815
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022248, 0.0021851
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005544, 0.0005445
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0028855, 0.0029378
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013372, 0.0013133
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005585, 0.0005686
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036291, 0.0036950
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009378, 0.0009211
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024264, 0.0023832
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012760, 0.0012533
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014533, 0.0014796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015910, upper bound: 0.0015649
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016503, upper bound: 0.0015089
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0028893, 0.0030670
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007199, 0.0007642
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0040500, 0.0038153
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0017365, 0.0018434
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0007839, 0.0007384
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0050938, 0.0047986
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0012179, 0.0012929
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0031512, 0.0033450
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0016572, 0.0017591
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0020398, 0.0019216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012374, upper bound: 0.0013464
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012374, upper bound: 0.0013464
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0030249, 0.0029285
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007537, 0.0007297
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0038671, 0.0039943
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0018180, 0.0017601
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0007485, 0.0007731
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0048638, 0.0050238
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0012751, 0.0012345
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0032991, 0.0031940
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0017349, 0.0016797
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0019477, 0.0020118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013285, upper bound: 0.0012489
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013285, upper bound: 0.0012489
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0029374, 0.0030251
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0007319, 0.0007538
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0039946, 0.0038789
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0017655, 0.0018182
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0007731, 0.0007507
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0050241, 0.0048786
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0012382, 0.0012752
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0032037, 0.0032993
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0016848, 0.0017351
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0020119, 0.0019536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015239, upper bound: 0.0016265
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015386, upper bound: 0.0016205
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0024044, 0.0024172
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005991, 0.0006023
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031919, 0.0031750
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014451, 0.0014528
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006178, 0.0006145
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0040145, 0.0039933
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010135, 0.0010189
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0026223, 0.0026363
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013790, 0.0013864
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0016076, 0.0015991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015838, upper bound: 0.0015430
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014927, upper bound: 0.0016610
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0023913, 0.0024374
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005959, 0.0006073
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0032186, 0.0031577
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014373, 0.0014650
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006230, 0.0006112
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0040482, 0.0039716
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010080, 0.0010275
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0026081, 0.0026584
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013716, 0.0013980
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0016211, 0.0015904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0014651
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0014651
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0024478, 0.0023720
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006099, 0.0005910
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031322, 0.0032323
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014712, 0.0014256
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006062, 0.0006256
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0039395, 0.0040654
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010318, 0.0009999
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0026697, 0.0025870
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014039, 0.0013605
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015775, 0.0016279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016473, upper bound: 0.0014947
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015392, upper bound: 0.0016058
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0024388, 0.0023976
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006077, 0.0005974
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031660, 0.0032204
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014658, 0.0014410
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006128, 0.0006233
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0039820, 0.0040504
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010280, 0.0010107
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0026598, 0.0026149
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013988, 0.0013752
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015946, 0.0016219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014891, upper bound: 0.0014407
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014891, upper bound: 0.0014407
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0024912, 0.0025777
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006208, 0.0006423
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0034039, 0.0032897
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014973, 0.0015493
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006588, 0.0006367
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0042812, 0.0041375
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010501, 0.0010866
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027170, 0.0028114
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014289, 0.0014785
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017144, 0.0016568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014861, upper bound: 0.0016626
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015051, upper bound: 0.0016476
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025133, 0.0025617
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006262, 0.0006383
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0033827, 0.0033188
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015106, 0.0015397
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006547, 0.0006423
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0042546, 0.0041742
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010594, 0.0010799
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027411, 0.0027939
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014415, 0.0014693
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017037, 0.0016715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013289, upper bound: 0.0013957
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013289, upper bound: 0.0013957
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025620, 0.0025052
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006384, 0.0006242
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0033080, 0.0033830
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015398, 0.0015057
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006403, 0.0006548
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0041606, 0.0042550
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010800, 0.0010560
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027942, 0.0027322
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014694, 0.0014369
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0016661, 0.0017039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0016058, upper bound: 0.0015392
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016265, upper bound: 0.0015239
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025829, 0.0024812
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006436, 0.0006182
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0032764, 0.0034107
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015524, 0.0014913
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006341, 0.0006601
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0041208, 0.0042897
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010888, 0.0010459
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0028170, 0.0027061
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014814, 0.0014231
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0016501, 0.0017178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0016186, upper bound: 0.0015220
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016342, upper bound: 0.0015039
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025375, 0.0025266
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006323, 0.0006296
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0033364, 0.0033508
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015251, 0.0015186
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006458, 0.0006485
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0041963, 0.0042144
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010697, 0.0010651
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027675, 0.0027557
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014554, 0.0014492
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0016804, 0.0016876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015736, upper bound: 0.0015601
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015922, upper bound: 0.0015475
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025551, 0.0025017
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006367, 0.0006233
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0033034, 0.0033740
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015357, 0.0015036
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006394, 0.0006530
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0041548, 0.0042436
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010771, 0.0010545
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027867, 0.0027284
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014655, 0.0014348
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0016638, 0.0016993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015820, upper bound: 0.0015394
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015989, upper bound: 0.0015213
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0023837, 0.0022737
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005940, 0.0005665
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0030023, 0.0031477
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014327, 0.0013665
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005811, 0.0006092
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0037762, 0.0039590
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010048, 0.0009584
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0025998, 0.0024797
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013672, 0.0013041
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015121, 0.0015853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016609, upper bound: 0.0014927
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016437, upper bound: 0.0015212
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0023998, 0.0022576
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005980, 0.0005625
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0029811, 0.0031688
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014423, 0.0013569
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005770, 0.0006133
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0037494, 0.0039856
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010116, 0.0009516
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0026173, 0.0024622
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013764, 0.0012948
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015014, 0.0015960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013804, upper bound: 0.0012768
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013804, upper bound: 0.0012768
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0023982, 0.0021988
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005976, 0.0005479
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0029035, 0.0031667
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014414, 0.0013215
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005620, 0.0006129
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036518, 0.0039829
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010109, 0.0009269
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0026155, 0.0023981
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013755, 0.0012611
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014623, 0.0015949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016818, upper bound: 0.0014777
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016985, upper bound: 0.0014626
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0023740, 0.0022251
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005915, 0.0005544
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0029382, 0.0031348
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014268, 0.0013373
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005687, 0.0006067
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036955, 0.0039428
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010007, 0.0009380
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0025892, 0.0024268
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013616, 0.0012762
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014798, 0.0015789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013458, upper bound: 0.0012516
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013458, upper bound: 0.0012516
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0021857, 0.0022425
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005446, 0.0005588
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0029612, 0.0028862
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013137, 0.0013478
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005731, 0.0005586
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0037244, 0.0036300
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009213, 0.0009453
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0023838, 0.0024457
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012536, 0.0012862
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014914, 0.0014536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015175, upper bound: 0.0016369
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015743, upper bound: 0.0015799
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0021659, 0.0022781
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005397, 0.0005676
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0030082, 0.0028600
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013017, 0.0013692
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005822, 0.0005535
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0037835, 0.0035971
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009130, 0.0009603
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0023622, 0.0024846
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012422, 0.0013066
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015151, 0.0014404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014972, upper bound: 0.0016712
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015641, upper bound: 0.0016205
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022871, 0.0023690
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005699, 0.0005903
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031282, 0.0030201
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013746, 0.0014238
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006055, 0.0005845
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0039345, 0.0037985
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009641, 0.0009986
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024944, 0.0025837
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013118, 0.0013587
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015755, 0.0015211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012979, upper bound: 0.0013530
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012979, upper bound: 0.0013530
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0023345, 0.0023222
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005817, 0.0005786
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0030665, 0.0030827
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014031, 0.0013957
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005935, 0.0005967
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0038568, 0.0038772
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009841, 0.0009789
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0025461, 0.0025327
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013390, 0.0013319
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015444, 0.0015526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013292, upper bound: 0.0013261
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013292, upper bound: 0.0013260
time: 1.17 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.43 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0013412, upper bound: 0.0013139
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0013412, upper bound: 0.0013139
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0013704, upper bound: 0.0012865
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0013704, upper bound: 0.0012865
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0012768, upper bound: 0.0013804
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0012768, upper bound: 0.0013804
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0012986, upper bound: 0.0013437
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0012986, upper bound: 0.0013437
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0016265, upper bound: 0.0015467
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0016790, upper bound: 0.0014815
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0015910, upper bound: 0.0015649
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0016503, upper bound: 0.0015089
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0012374, upper bound: 0.0013464
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0012374, upper bound: 0.0013464
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0013285, upper bound: 0.0012489
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0013285, upper bound: 0.0012489
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0015239, upper bound: 0.0016265
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0015386, upper bound: 0.0016205
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0015838, upper bound: 0.0015430
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0014927, upper bound: 0.0016610
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0014651
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0014651
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0016473, upper bound: 0.0014947
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0015392, upper bound: 0.0016058
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0014891, upper bound: 0.0014407
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0014891, upper bound: 0.0014407
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0014861, upper bound: 0.0016626
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0015051, upper bound: 0.0016476
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0013289, upper bound: 0.0013957
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0013289, upper bound: 0.0013957
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0016058, upper bound: 0.0015392
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0016265, upper bound: 0.0015239
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0016186, upper bound: 0.0015220
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0016342, upper bound: 0.0015039
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0015736, upper bound: 0.0015601
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0015922, upper bound: 0.0015475
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0015820, upper bound: 0.0015394
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0015989, upper bound: 0.0015213
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0016609, upper bound: 0.0014927
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0016437, upper bound: 0.0015212
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0013804, upper bound: 0.0012768
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0013804, upper bound: 0.0012768
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0016818, upper bound: 0.0014777
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0016985, upper bound: 0.0014626
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0013458, upper bound: 0.0012516
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0013458, upper bound: 0.0012516
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0015175, upper bound: 0.0016369
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0015743, upper bound: 0.0015799
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0014972, upper bound: 0.0016712
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0015641, upper bound: 0.0016205
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0012979, upper bound: 0.0013530
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0012979, upper bound: 0.0013530
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0013292, upper bound: 0.0013261
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 0, lower bound: -0.0013292, upper bound: 0.0013260

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0021638, 0.0021161
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005392, 0.0005273
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0027943, 0.0028573
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013005, 0.0012718
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005408, 0.0005530
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0035145, 0.0035937
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009121, 0.0008920
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0023599, 0.0023079
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012411, 0.0012137
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014074, 0.0014391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013112, upper bound: 0.0012526
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013112, upper bound: 0.0012526
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022063, 0.0020678
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005498, 0.0005153
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0027306, 0.0029134
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013261, 0.0012428
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005285, 0.0005639
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0034343, 0.0036643
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009300, 0.0008717
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024063, 0.0022553
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012655, 0.0011860
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0013753, 0.0014674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013390, upper bound: 0.0012241
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013390, upper bound: 0.0012241
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0021822, 0.0020933
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005437, 0.0005216
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0027641, 0.0028815
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013115, 0.0012581
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005350, 0.0005577
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0034766, 0.0036242
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009199, 0.0008824
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0023800, 0.0022830
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012516, 0.0012006
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0013922, 0.0014513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013238, upper bound: 0.0012429
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013238, upper bound: 0.0012429
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0021038, 0.0021706
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005242, 0.0005408
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0028662, 0.0027780
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0012644, 0.0013046
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005547, 0.0005377
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036049, 0.0034940
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0008868, 0.0009150
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0022944, 0.0023673
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012066, 0.0012449
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014436, 0.0013991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012526, upper bound: 0.0013112
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012526, upper bound: 0.0013112
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0020877, 0.0021914
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005202, 0.0005460
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0028937, 0.0027567
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0012548, 0.0013171
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005601, 0.0005336
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036395, 0.0034673
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0008800, 0.0009237
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0022769, 0.0023900
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0011974, 0.0012569
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014574, 0.0013884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012612, upper bound: 0.0013045
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012612, upper bound: 0.0013045
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0020767, 0.0021868
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005175, 0.0005449
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0028876, 0.0027423
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0012482, 0.0013143
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005589, 0.0005308
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036318, 0.0034491
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0008754, 0.0009218
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0022650, 0.0023850
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0011911, 0.0012542
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014543, 0.0013812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012341, upper bound: 0.0013214
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012341, upper bound: 0.0013214
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022133, 0.0020443
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005515, 0.0005094
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0026995, 0.0029227
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013303, 0.0012287
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005225, 0.0005657
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0033953, 0.0036759
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009330, 0.0008618
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024139, 0.0022296
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012695, 0.0011725
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0013596, 0.0014720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013232, upper bound: 0.0012276
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013232, upper bound: 0.0012276
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0020893, 0.0021985
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005206, 0.0005478
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0029031, 0.0027588
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0012557, 0.0013214
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005619, 0.0005340
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036514, 0.0034699
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0008807, 0.0009268
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0022786, 0.0023978
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0011983, 0.0012610
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014622, 0.0013895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012219, upper bound: 0.0013367
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012219, upper bound: 0.0013367
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0021159, 0.0021763
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005272, 0.0005423
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0028738, 0.0027941
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0012717, 0.0013080
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005562, 0.0005408
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036145, 0.0035142
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0008919, 0.0009174
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0023077, 0.0023736
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012136, 0.0012483
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014474, 0.0014072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012356, upper bound: 0.0013215
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012356, upper bound: 0.0013215
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0021914, 0.0021038
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005460, 0.0005242
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0027780, 0.0028937
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013171, 0.0012644
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005377, 0.0005601
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0034940, 0.0036396
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009238, 0.0008868
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0023901, 0.0022944
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012569, 0.0012066
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0013991, 0.0014574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013112, upper bound: 0.0012526
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013112, upper bound: 0.0012526
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022071, 0.0020798
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005499, 0.0005182
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0027463, 0.0029144
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013265, 0.0012500
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005315, 0.0005641
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0034541, 0.0036656
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009304, 0.0008767
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024071, 0.0022683
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012659, 0.0011929
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0013832, 0.0014679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013195, upper bound: 0.0012298
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013195, upper bound: 0.0012298
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022062, 0.0020767
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005497, 0.0005175
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0027423, 0.0029132
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013260, 0.0012482
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005308, 0.0005638
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0034491, 0.0036641
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009300, 0.0008754
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024061, 0.0022650
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012654, 0.0011911
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0013812, 0.0014673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013214, upper bound: 0.0012341
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013214, upper bound: 0.0012341
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0021879, 0.0021088
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005452, 0.0005255
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0027847, 0.0028891
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013150, 0.0012675
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005390, 0.0005592
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0035024, 0.0036338
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009223, 0.0008889
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0023863, 0.0023000
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012549, 0.0012095
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014025, 0.0014551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013058, upper bound: 0.0012546
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013058, upper bound: 0.0012546
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022339, 0.0020556
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005566, 0.0005122
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0027144, 0.0029499
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013427, 0.0012355
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005254, 0.0005709
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0034140, 0.0037102
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009417, 0.0008665
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024364, 0.0022419
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012813, 0.0011790
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0013671, 0.0014857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013390, upper bound: 0.0012241
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013390, upper bound: 0.0012241
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022526, 0.0020353
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005613, 0.0005072
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0026876, 0.0029745
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013539, 0.0012233
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005202, 0.0005757
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0033803, 0.0037411
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009495, 0.0008580
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024567, 0.0022198
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012920, 0.0011674
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0013536, 0.0014981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013554, upper bound: 0.0012083
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013554, upper bound: 0.0012083
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0020867, 0.0021958
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005200, 0.0005471
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0028995, 0.0027555
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0012542, 0.0013197
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005612, 0.0005333
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036469, 0.0034657
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0008796, 0.0009256
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0022759, 0.0023948
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0011969, 0.0012594
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014604, 0.0013878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012463, upper bound: 0.0013113
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012463, upper bound: 0.0013113
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0020669, 0.0022265
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005150, 0.0005548
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0029401, 0.0027294
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0012423, 0.0013382
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005691, 0.0005283
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036979, 0.0034328
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0008713, 0.0009386
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0022543, 0.0024283
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0011855, 0.0012770
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014808, 0.0013747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012306, upper bound: 0.0013316
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012306, upper bound: 0.0013316
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0021153, 0.0021862
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005271, 0.0005447
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0028869, 0.0027932
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0012713, 0.0013140
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005587, 0.0005406
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036309, 0.0035131
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0008917, 0.0009216
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0023070, 0.0023844
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012132, 0.0012539
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014540, 0.0014068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012612, upper bound: 0.0013045
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012612, upper bound: 0.0013045
time: 1.10 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.36 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013112, upper bound: 0.0012526
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013112, upper bound: 0.0012526
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013390, upper bound: 0.0012241
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013390, upper bound: 0.0012241
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013238, upper bound: 0.0012429
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013238, upper bound: 0.0012429
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012526, upper bound: 0.0013112
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012526, upper bound: 0.0013112
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012612, upper bound: 0.0013045
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012612, upper bound: 0.0013045
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012341, upper bound: 0.0013214
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012341, upper bound: 0.0013214
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013232, upper bound: 0.0012276
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013232, upper bound: 0.0012276
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012219, upper bound: 0.0013367
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012219, upper bound: 0.0013367
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012356, upper bound: 0.0013215
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012356, upper bound: 0.0013215
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013112, upper bound: 0.0012526
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013112, upper bound: 0.0012526
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013195, upper bound: 0.0012298
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013195, upper bound: 0.0012298
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013214, upper bound: 0.0012341
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013214, upper bound: 0.0012341
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013058, upper bound: 0.0012546
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013058, upper bound: 0.0012546
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013390, upper bound: 0.0012241
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013390, upper bound: 0.0012241
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013554, upper bound: 0.0012083
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013554, upper bound: 0.0012083
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012463, upper bound: 0.0013113
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012463, upper bound: 0.0013113
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012306, upper bound: 0.0013316
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012306, upper bound: 0.0013316
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012612, upper bound: 0.0013045
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0012612, upper bound: 0.0013045

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.26 + 347.95 = 351.21 seconds
