## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00603328


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0031114, -0.0018525, -0.0031114, -0.0018525, -0.0009866, 0.0009866)
1: (0.0228612, 0.0288143, 0.0228612, 0.0288143, -0.0030931, 0.0030931)
2: (0.0226511, 0.0267366, 0.0226511, 0.0267366, -0.0022643, 0.0022643)
3: (0.0103305, 0.0149545, 0.0103305, 0.0149545, -0.0032307, 0.0032307)
4: (-0.0153141, -0.0103896, -0.0153141, -0.0103896, -0.0034332, 0.0034332)
5: (0.0174743, 0.0231114, 0.0174743, 0.0231114, -0.0038617, 0.0038617)
6: (0.0083046, 0.0127426, 0.0083046, 0.0127426, -0.0031681, 0.0031681)
7: (-0.0200006, -0.0154148, -0.0200006, -0.0154148, -0.0030386, 0.0030386)
8: (0.0123498, 0.0167545, 0.0123498, 0.0167545, -0.0030039, 0.0030039)
9: (0.9117882, 0.9334683, 0.9117882, 0.9334683, -0.0141678, 0.0141677)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.30 + 1.31 = 2.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0078645, upper bound: 0.0078645

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0076287, upper bound: 0.0076215
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0076215, upper bound: 0.0076287
time: 0.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.12 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.12
Output dim: 9, lower bound: -0.0076287, upper bound: 0.0076215
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.12
Output dim: 9, lower bound: -0.0076215, upper bound: 0.0076287

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0031114, -0.0018525, -0.0031114, -0.0018525, -0.0009817, 0.0009809
1: 0.0228612, 0.0288143, 0.0228612, 0.0288143, -0.0028583, 0.0028465
2: 0.0226511, 0.0267366, 0.0226511, 0.0267366, -0.0021627, 0.0021557
3: 0.0103305, 0.0149545, 0.0103305, 0.0149545, -0.0030348, 0.0030315
4: -0.0153141, -0.0103896, -0.0153141, -0.0103896, -0.0033019, 0.0033120
5: 0.0174743, 0.0231114, 0.0174743, 0.0231114, -0.0036978, 0.0036884
6: 0.0083046, 0.0127426, 0.0083046, 0.0127426, -0.0030321, 0.0030248
7: -0.0200006, -0.0154148, -0.0200006, -0.0154148, -0.0029290, 0.0029359
8: 0.0123498, 0.0167545, 0.0123498, 0.0167545, -0.0028066, 0.0028064
9: 0.9117882, 0.9334683, 0.9117882, 0.9334683, -0.0133728, 0.0133945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0064663, upper bound: 0.0069823
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0069864, upper bound: 0.0064663
time: 0.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0031114, -0.0018525, -0.0031114, -0.0018525, -0.0009809, 0.0009817
1: 0.0228612, 0.0288143, 0.0228612, 0.0288143, -0.0028465, 0.0028583
2: 0.0226511, 0.0267366, 0.0226511, 0.0267366, -0.0021557, 0.0021627
3: 0.0103305, 0.0149545, 0.0103305, 0.0149545, -0.0030315, 0.0030348
4: -0.0153141, -0.0103896, -0.0153141, -0.0103896, -0.0033120, 0.0033019
5: 0.0174743, 0.0231114, 0.0174743, 0.0231114, -0.0036884, 0.0036978
6: 0.0083046, 0.0127426, 0.0083046, 0.0127426, -0.0030248, 0.0030321
7: -0.0200006, -0.0154148, -0.0200006, -0.0154148, -0.0029359, 0.0029290
8: 0.0123498, 0.0167545, 0.0123498, 0.0167545, -0.0028064, 0.0028066
9: 0.9117882, 0.9334683, 0.9117882, 0.9334683, -0.0133945, 0.0133728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0064663, upper bound: 0.0069864
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0069823, upper bound: 0.0064663
time: 0.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 9, lower bound: -0.0064663, upper bound: 0.0069823
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 9, lower bound: -0.0069864, upper bound: 0.0064663
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 9, lower bound: -0.0064663, upper bound: 0.0069864
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 9, lower bound: -0.0069823, upper bound: 0.0064663

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0031114, -0.0018525, -0.0031114, -0.0018525, -0.0009228, 0.0009039
1: 0.0228612, 0.0288143, 0.0228612, 0.0288143, -0.0024196, 0.0022903
2: 0.0226511, 0.0267366, 0.0226511, 0.0267366, -0.0018519, 0.0017434
3: 0.0103305, 0.0149545, 0.0103305, 0.0149545, -0.0026832, 0.0025300
4: -0.0153141, -0.0103896, -0.0153141, -0.0103896, -0.0027726, 0.0029643
5: 0.0174743, 0.0231114, 0.0174743, 0.0231114, -0.0032910, 0.0030826
6: 0.0083046, 0.0127426, 0.0083046, 0.0127426, -0.0027060, 0.0025403
7: -0.0200006, -0.0154148, -0.0200006, -0.0154148, -0.0024427, 0.0026068
8: 0.0123498, 0.0167545, 0.0123498, 0.0167545, -0.0024802, 0.0023346
9: 0.9117882, 0.9334683, 0.9117882, 0.9334683, -0.0110623, 0.0117581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0063652, upper bound: 0.0069085
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0063892, upper bound: 0.0067882
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0031114, -0.0018525, -0.0031114, -0.0018525, -0.0009047, 0.0009223
1: 0.0228612, 0.0288143, 0.0228612, 0.0288143, -0.0023021, 0.0024200
2: 0.0226511, 0.0267366, 0.0226511, 0.0267366, -0.0017504, 0.0018515
3: 0.0103305, 0.0149545, 0.0103305, 0.0149545, -0.0025333, 0.0026857
4: -0.0153141, -0.0103896, -0.0153141, -0.0103896, -0.0029587, 0.0027827
5: 0.0174743, 0.0231114, 0.0174743, 0.0231114, -0.0030919, 0.0032868
6: 0.0083046, 0.0127426, 0.0083046, 0.0127426, -0.0025476, 0.0027045
7: -0.0200006, -0.0154148, -0.0200006, -0.0154148, -0.0026006, 0.0024496
8: 0.0123498, 0.0167545, 0.0123498, 0.0167545, -0.0023348, 0.0024823
9: 0.9117882, 0.9334683, 0.9117882, 0.9334683, -0.0117646, 0.0110840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0067932, upper bound: 0.0063892
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0069125, upper bound: 0.0063673
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0031114, -0.0018525, -0.0031114, -0.0018525, -0.0009223, 0.0009047
1: 0.0228612, 0.0288143, 0.0228612, 0.0288143, -0.0024200, 0.0023020
2: 0.0226511, 0.0267366, 0.0226511, 0.0267366, -0.0018515, 0.0017504
3: 0.0103305, 0.0149545, 0.0103305, 0.0149545, -0.0026857, 0.0025333
4: -0.0153141, -0.0103896, -0.0153141, -0.0103896, -0.0027827, 0.0029587
5: 0.0174743, 0.0231114, 0.0174743, 0.0231114, -0.0032868, 0.0030919
6: 0.0083046, 0.0127426, 0.0083046, 0.0127426, -0.0027045, 0.0025476
7: -0.0200006, -0.0154148, -0.0200006, -0.0154148, -0.0024496, 0.0026006
8: 0.0123498, 0.0167545, 0.0123498, 0.0167545, -0.0024823, 0.0023348
9: 0.9117882, 0.9334683, 0.9117882, 0.9334683, -0.0110840, 0.0117646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0063673, upper bound: 0.0069125
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0063892, upper bound: 0.0067932
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0031114, -0.0018525, -0.0031114, -0.0018525, -0.0009039, 0.0009228
1: 0.0228612, 0.0288143, 0.0228612, 0.0288143, -0.0022903, 0.0024196
2: 0.0226511, 0.0267366, 0.0226511, 0.0267366, -0.0017434, 0.0018519
3: 0.0103305, 0.0149545, 0.0103305, 0.0149545, -0.0025300, 0.0026832
4: -0.0153141, -0.0103896, -0.0153141, -0.0103896, -0.0029643, 0.0027726
5: 0.0174743, 0.0231114, 0.0174743, 0.0231114, -0.0030826, 0.0032910
6: 0.0083046, 0.0127426, 0.0083046, 0.0127426, -0.0025403, 0.0027060
7: -0.0200006, -0.0154148, -0.0200006, -0.0154148, -0.0026068, 0.0024427
8: 0.0123498, 0.0167545, 0.0123498, 0.0167545, -0.0023346, 0.0024802
9: 0.9117882, 0.9334683, 0.9117882, 0.9334683, -0.0117581, 0.0110623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0067882, upper bound: 0.0063892
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0069085, upper bound: 0.0063652
time: 0.53 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.49 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 9, lower bound: -0.0063652, upper bound: 0.0069085
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 9, lower bound: -0.0063892, upper bound: 0.0067882
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 9, lower bound: -0.0067932, upper bound: 0.0063892
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 9, lower bound: -0.0069125, upper bound: 0.0063673
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 9, lower bound: -0.0063673, upper bound: 0.0069125
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 9, lower bound: -0.0063892, upper bound: 0.0067932
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 9, lower bound: -0.0067882, upper bound: 0.0063892
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 9, lower bound: -0.0069085, upper bound: 0.0063652

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0031114, -0.0018525, -0.0031114, -0.0018525, -0.0008621, 0.0008330
1: 0.0228612, 0.0288143, 0.0228612, 0.0288143, -0.0022792, 0.0021253
2: 0.0226511, 0.0267366, 0.0226511, 0.0267366, -0.0016962, 0.0015556
3: 0.0103305, 0.0149545, 0.0103305, 0.0149545, -0.0024855, 0.0022897
4: -0.0153141, -0.0103896, -0.0153141, -0.0103896, -0.0024407, 0.0026881
5: 0.0174743, 0.0231114, 0.0174743, 0.0231114, -0.0029855, 0.0027151
6: 0.0083046, 0.0127426, 0.0083046, 0.0127426, -0.0024720, 0.0022576
7: -0.0200006, -0.0154148, -0.0200006, -0.0154148, -0.0021715, 0.0023879
8: 0.0123498, 0.0167545, 0.0123498, 0.0167545, -0.0022864, 0.0021103
9: 0.9117882, 0.9334683, 0.9117882, 0.9334683, -0.0098885, 0.0107977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024472, upper bound: 0.0024841
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024472, upper bound: 0.0024841
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0031114, -0.0018525, -0.0031114, -0.0018525, -0.0008383, 0.0008431
1: 0.0228612, 0.0288143, 0.0228612, 0.0288143, -0.0022352, 0.0021500
2: 0.0226511, 0.0267366, 0.0226511, 0.0267366, -0.0016492, 0.0015878
3: 0.0103305, 0.0149545, 0.0103305, 0.0149545, -0.0024071, 0.0023324
4: -0.0153141, -0.0103896, -0.0153141, -0.0103896, -0.0024964, 0.0025877
5: 0.0174743, 0.0231114, 0.0174743, 0.0231114, -0.0028801, 0.0027771
6: 0.0083046, 0.0127426, 0.0083046, 0.0127426, -0.0023880, 0.0023063
7: -0.0200006, -0.0154148, -0.0200006, -0.0154148, -0.0022238, 0.0023003
8: 0.0123498, 0.0167545, 0.0123498, 0.0167545, -0.0022135, 0.0021408
9: 0.9117882, 0.9334683, 0.9117882, 0.9334683, -0.0101019, 0.0104375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024600, upper bound: 0.0024580
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024600, upper bound: 0.0024580
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0031114, -0.0018525, -0.0031114, -0.0018525, -0.0008439, 0.0008381
1: 0.0228612, 0.0288143, 0.0228612, 0.0288143, -0.0021617, 0.0022344
2: 0.0226511, 0.0267366, 0.0226511, 0.0267366, -0.0015948, 0.0016476
3: 0.0103305, 0.0149545, 0.0103305, 0.0149545, -0.0023356, 0.0024094
4: -0.0153141, -0.0103896, -0.0153141, -0.0103896, -0.0025864, 0.0025065
5: 0.0174743, 0.0231114, 0.0174743, 0.0231114, -0.0027865, 0.0028798
6: 0.0083046, 0.0127426, 0.0083046, 0.0127426, -0.0023136, 0.0023886
7: -0.0200006, -0.0154148, -0.0200006, -0.0154148, -0.0022981, 0.0022307
8: 0.0123498, 0.0167545, 0.0123498, 0.0167545, -0.0021410, 0.0022161
9: 0.9117882, 0.9334683, 0.9117882, 0.9334683, -0.0104365, 0.0101237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024580, upper bound: 0.0024600
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024580, upper bound: 0.0024600
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0031114, -0.0018525, -0.0031114, -0.0018525, -0.0008334, 0.0008616
1: 0.0228612, 0.0288143, 0.0228612, 0.0288143, -0.0021382, 0.0022797
2: 0.0226511, 0.0267366, 0.0226511, 0.0267366, -0.0015627, 0.0016959
3: 0.0103305, 0.0149545, 0.0103305, 0.0149545, -0.0022882, 0.0024881
4: -0.0153141, -0.0103896, -0.0153141, -0.0103896, -0.0026825, 0.0024439
5: 0.0174743, 0.0231114, 0.0174743, 0.0231114, -0.0027176, 0.0029814
6: 0.0083046, 0.0127426, 0.0083046, 0.0127426, -0.0022595, 0.0024705
7: -0.0200006, -0.0154148, -0.0200006, -0.0154148, -0.0023817, 0.0021741
8: 0.0123498, 0.0167545, 0.0123498, 0.0167545, -0.0021088, 0.0022884
9: 0.9117882, 0.9334683, 0.9117882, 0.9334683, -0.0108043, 0.0099005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024841, upper bound: 0.0024472
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024841, upper bound: 0.0024472
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0031114, -0.0018525, -0.0031114, -0.0018525, -0.0008616, 0.0008334
1: 0.0228612, 0.0288143, 0.0228612, 0.0288143, -0.0022797, 0.0021382
2: 0.0226511, 0.0267366, 0.0226511, 0.0267366, -0.0016959, 0.0015627
3: 0.0103305, 0.0149545, 0.0103305, 0.0149545, -0.0024881, 0.0022882
4: -0.0153141, -0.0103896, -0.0153141, -0.0103896, -0.0024439, 0.0026825
5: 0.0174743, 0.0231114, 0.0174743, 0.0231114, -0.0029814, 0.0027176
6: 0.0083046, 0.0127426, 0.0083046, 0.0127426, -0.0024705, 0.0022595
7: -0.0200006, -0.0154148, -0.0200006, -0.0154148, -0.0021741, 0.0023817
8: 0.0123498, 0.0167545, 0.0123498, 0.0167545, -0.0022884, 0.0021088
9: 0.9117882, 0.9334683, 0.9117882, 0.9334683, -0.0099005, 0.0108043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024472, upper bound: 0.0024841
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024472, upper bound: 0.0024841
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0031114, -0.0018525, -0.0031114, -0.0018525, -0.0008381, 0.0008439
1: 0.0228612, 0.0288143, 0.0228612, 0.0288143, -0.0022344, 0.0021617
2: 0.0226511, 0.0267366, 0.0226511, 0.0267366, -0.0016476, 0.0015948
3: 0.0103305, 0.0149545, 0.0103305, 0.0149545, -0.0024094, 0.0023356
4: -0.0153141, -0.0103896, -0.0153141, -0.0103896, -0.0025065, 0.0025864
5: 0.0174743, 0.0231114, 0.0174743, 0.0231114, -0.0028798, 0.0027865
6: 0.0083046, 0.0127426, 0.0083046, 0.0127426, -0.0023886, 0.0023136
7: -0.0200006, -0.0154148, -0.0200006, -0.0154148, -0.0022307, 0.0022982
8: 0.0123498, 0.0167545, 0.0123498, 0.0167545, -0.0022161, 0.0021410
9: 0.9117882, 0.9334683, 0.9117882, 0.9334683, -0.0101237, 0.0104365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024600, upper bound: 0.0024580
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024600, upper bound: 0.0024580
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0031114, -0.0018525, -0.0031114, -0.0018525, -0.0008431, 0.0008383
1: 0.0228612, 0.0288143, 0.0228612, 0.0288143, -0.0021500, 0.0022352
2: 0.0226511, 0.0267366, 0.0226511, 0.0267366, -0.0015878, 0.0016492
3: 0.0103305, 0.0149545, 0.0103305, 0.0149545, -0.0023324, 0.0024071
4: -0.0153141, -0.0103896, -0.0153141, -0.0103896, -0.0025877, 0.0024964
5: 0.0174743, 0.0231114, 0.0174743, 0.0231114, -0.0027771, 0.0028801
6: 0.0083046, 0.0127426, 0.0083046, 0.0127426, -0.0023063, 0.0023880
7: -0.0200006, -0.0154148, -0.0200006, -0.0154148, -0.0023003, 0.0022238
8: 0.0123498, 0.0167545, 0.0123498, 0.0167545, -0.0021408, 0.0022135
9: 0.9117882, 0.9334683, 0.9117882, 0.9334683, -0.0104375, 0.0101019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024580, upper bound: 0.0024600
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024580, upper bound: 0.0024600
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0031114, -0.0018525, -0.0031114, -0.0018525, -0.0008330, 0.0008621
1: 0.0228612, 0.0288143, 0.0228612, 0.0288143, -0.0021253, 0.0022792
2: 0.0226511, 0.0267366, 0.0226511, 0.0267366, -0.0015556, 0.0016962
3: 0.0103305, 0.0149545, 0.0103305, 0.0149545, -0.0022897, 0.0024855
4: -0.0153141, -0.0103896, -0.0153141, -0.0103896, -0.0026881, 0.0024407
5: 0.0174743, 0.0231114, 0.0174743, 0.0231114, -0.0027151, 0.0029855
6: 0.0083046, 0.0127426, 0.0083046, 0.0127426, -0.0022576, 0.0024720
7: -0.0200006, -0.0154148, -0.0200006, -0.0154148, -0.0023879, 0.0021715
8: 0.0123498, 0.0167545, 0.0123498, 0.0167545, -0.0021103, 0.0022864
9: 0.9117882, 0.9334683, 0.9117882, 0.9334683, -0.0107977, 0.0098885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024841, upper bound: 0.0024472
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024841, upper bound: 0.0024472
time: 0.46 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024472, upper bound: 0.0024841
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024472, upper bound: 0.0024841
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024600, upper bound: 0.0024580
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024600, upper bound: 0.0024580
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024580, upper bound: 0.0024600
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024580, upper bound: 0.0024600
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024841, upper bound: 0.0024472
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024841, upper bound: 0.0024472
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024472, upper bound: 0.0024841
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024472, upper bound: 0.0024841
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024600, upper bound: 0.0024580
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024600, upper bound: 0.0024580
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024580, upper bound: 0.0024600
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024580, upper bound: 0.0024600
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024841, upper bound: 0.0024472
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 9, lower bound: -0.0024841, upper bound: 0.0024472

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.60 + 33.88 = 36.49 seconds
