## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00085666


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0020365, 0.0020365)
1: (-0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0005075, 0.0005075)
2: (0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026892, 0.0026892)
3: (-0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0012240, 0.0012240)
4: (0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005205, 0.0005205)
5: (0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0033824, 0.0033824)
6: (0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008585, 0.0008585)
7: (-0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0022211, 0.0022211)
8: (-0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011681, 0.0011681)
9: (-0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013544, 0.0013544)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.57 + 1.49 = 3.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0012182, upper bound: 0.0012183

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011898, upper bound: 0.0011913
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011913, upper bound: 0.0011898
time: 0.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.39 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.39
Output dim: 0, lower bound: -0.0011898, upper bound: 0.0011913
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.39
Output dim: 0, lower bound: -0.0011913, upper bound: 0.0011898

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019852, 0.0019855
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004947, 0.0004947
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026218, 0.0026215
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011932, 0.0011933
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005074, 0.0005074
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032976, 0.0032972
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008369, 0.0008370
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021652, 0.0021655
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011387, 0.0011388
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013205, 0.0013203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011831, upper bound: 0.0011558
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011514, upper bound: 0.0011847
time: 0.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019855, 0.0019852
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004947, 0.0004947
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026215, 0.0026218
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011933, 0.0011932
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005074, 0.0005074
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032971, 0.0032976
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008370, 0.0008369
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021655, 0.0021652
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011388, 0.0011387
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013203, 0.0013205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011846, upper bound: 0.0011514
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011558, upper bound: 0.0011831
time: 0.69 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -0.0011831, upper bound: 0.0011558
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -0.0011514, upper bound: 0.0011847
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -0.0011846, upper bound: 0.0011514
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -0.0011558, upper bound: 0.0011831

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019846, 0.0019823
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004945, 0.0004939
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026177, 0.0026206
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011928, 0.0011914
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005066, 0.0005072
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032923, 0.0032961
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008366, 0.0008356
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021645, 0.0021620
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011383, 0.0011370
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013184, 0.0013199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011633, upper bound: 0.0011374
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011635, upper bound: 0.0011337
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019824, 0.0019848
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004939, 0.0004946
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026210, 0.0026177
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011915, 0.0011930
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005073, 0.0005066
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032965, 0.0032923
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008356, 0.0008367
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021620, 0.0021648
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011370, 0.0011384
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013201, 0.0013184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011304, upper bound: 0.0011650
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011326, upper bound: 0.0011649
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019848, 0.0019824
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004946, 0.0004939
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026177, 0.0026210
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011930, 0.0011915
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005066, 0.0005073
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032923, 0.0032965
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008367, 0.0008356
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021648, 0.0021620
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011384, 0.0011370
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013184, 0.0013201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011649, upper bound: 0.0011327
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011650, upper bound: 0.0011305
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019823, 0.0019846
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004939, 0.0004945
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026206, 0.0026177
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011914, 0.0011928
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005072, 0.0005066
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032961, 0.0032923
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008356, 0.0008366
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021620, 0.0021645
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011370, 0.0011383
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013199, 0.0013184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011337, upper bound: 0.0011635
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011373, upper bound: 0.0011633
time: 0.61 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.08 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0011633, upper bound: 0.0011374
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0011635, upper bound: 0.0011337
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0011304, upper bound: 0.0011650
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0011326, upper bound: 0.0011649
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0011649, upper bound: 0.0011327
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0011650, upper bound: 0.0011305
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0011337, upper bound: 0.0011635
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0011373, upper bound: 0.0011633

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019815, 0.0019787
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004937, 0.0004930
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026128, 0.0026166
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011909, 0.0011892
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005057, 0.0005064
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032862, 0.0032909
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008353, 0.0008341
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021611, 0.0021580
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011365, 0.0011349
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013160, 0.0013178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009452, upper bound: 0.0009262
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009452, upper bound: 0.0009262
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019818, 0.0019792
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004938, 0.0004932
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026136, 0.0026170
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011911, 0.0011896
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005059, 0.0005065
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032872, 0.0032914
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008354, 0.0008343
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021614, 0.0021586
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011367, 0.0011352
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013163, 0.0013180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009459, upper bound: 0.0009242
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009459, upper bound: 0.0009242
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019793, 0.0019814
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004932, 0.0004937
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026165, 0.0026136
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011896, 0.0011909
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005064, 0.0005059
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032908, 0.0032872
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008343, 0.0008352
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021587, 0.0021610
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011352, 0.0011365
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013178, 0.0013163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009240, upper bound: 0.0009459
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009240, upper bound: 0.0009459
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019795, 0.0019818
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004932, 0.0004938
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026169, 0.0026140
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011898, 0.0011911
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005065, 0.0005059
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032913, 0.0032877
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008344, 0.0008354
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021590, 0.0021614
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011354, 0.0011366
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013180, 0.0013165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009261, upper bound: 0.0009452
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009261, upper bound: 0.0009452
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019818, 0.0019795
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004938, 0.0004932
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026140, 0.0026169
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011911, 0.0011898
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005059, 0.0005065
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032877, 0.0032913
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008354, 0.0008344
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021614, 0.0021590
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011366, 0.0011354
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013165, 0.0013180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009452, upper bound: 0.0009261
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009452, upper bound: 0.0009261
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019814, 0.0019793
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004937, 0.0004932
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026136, 0.0026165
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011909, 0.0011896
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005059, 0.0005064
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032872, 0.0032908
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008352, 0.0008343
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021610, 0.0021587
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011365, 0.0011352
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013163, 0.0013178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009459, upper bound: 0.0009240
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009459, upper bound: 0.0009240
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019792, 0.0019818
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004932, 0.0004938
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026170, 0.0026136
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011896, 0.0011911
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005065, 0.0005059
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032914, 0.0032872
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008343, 0.0008354
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021586, 0.0021614
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011352, 0.0011367
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013180, 0.0013163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009242, upper bound: 0.0009459
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009242, upper bound: 0.0009459
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019787, 0.0019815
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004930, 0.0004937
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026166, 0.0026128
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011892, 0.0011909
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005064, 0.0005057
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032909, 0.0032862
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008341, 0.0008353
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021580, 0.0021611
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011349, 0.0011365
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013178, 0.0013160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009262, upper bound: 0.0009452
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009262, upper bound: 0.0009452
time: 0.63 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009452, upper bound: 0.0009262
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009452, upper bound: 0.0009262
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009459, upper bound: 0.0009242
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009459, upper bound: 0.0009242
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009240, upper bound: 0.0009459
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009240, upper bound: 0.0009459
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009261, upper bound: 0.0009452
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009261, upper bound: 0.0009452
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009452, upper bound: 0.0009261
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009452, upper bound: 0.0009261
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009459, upper bound: 0.0009240
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009459, upper bound: 0.0009240
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009242, upper bound: 0.0009459
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009242, upper bound: 0.0009459
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009262, upper bound: 0.0009452
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 0, lower bound: -0.0009262, upper bound: 0.0009452

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019744, 0.0019928
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004920, 0.0004965
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026314, 0.0026071
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011866, 0.0011977
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005093, 0.0005046
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0033096, 0.0032791
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008323, 0.0008400
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021533, 0.0021734
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011324, 0.0011430
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013253, 0.0013131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008544, upper bound: 0.0007950
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008071, upper bound: 0.0008387
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019815, 0.0019715
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004937, 0.0004913
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026034, 0.0026166
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011909, 0.0011849
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005039, 0.0005064
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032744, 0.0032909
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008353, 0.0008311
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021611, 0.0021502
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011365, 0.0011308
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013112, 0.0013178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008544, upper bound: 0.0007950
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008071, upper bound: 0.0008387
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019747, 0.0019916
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004920, 0.0004963
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026299, 0.0026075
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011868, 0.0011970
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005090, 0.0005047
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0033077, 0.0032796
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008324, 0.0008395
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021536, 0.0021721
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011326, 0.0011423
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013245, 0.0013133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008545, upper bound: 0.0007931
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008071, upper bound: 0.0008376
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019818, 0.0019721
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004938, 0.0004914
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026041, 0.0026170
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011911, 0.0011853
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005040, 0.0005065
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032753, 0.0032914
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008354, 0.0008313
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021614, 0.0021508
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011367, 0.0011311
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013116, 0.0013180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008545, upper bound: 0.0007931
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008071, upper bound: 0.0008376
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019721, 0.0019950
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004914, 0.0004971
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026344, 0.0026041
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011853, 0.0011991
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005099, 0.0005040
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0033133, 0.0032753
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008313, 0.0008410
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021509, 0.0021758
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011311, 0.0011442
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013268, 0.0013116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008375, upper bound: 0.0008080
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007925, upper bound: 0.0008546
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019793, 0.0019743
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004932, 0.0004919
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026070, 0.0026136
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011896, 0.0011866
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005046, 0.0005059
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032789, 0.0032872
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008343, 0.0008322
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021587, 0.0021532
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011352, 0.0011324
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013130, 0.0013163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008375, upper bound: 0.0008080
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007925, upper bound: 0.0008546
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019724, 0.0019938
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004915, 0.0004968
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026328, 0.0026045
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011855, 0.0011983
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005096, 0.0005041
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0033114, 0.0032758
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008314, 0.0008405
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021512, 0.0021745
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011313, 0.0011436
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013260, 0.0013118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008385, upper bound: 0.0008080
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007948, upper bound: 0.0008545
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019795, 0.0019746
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004932, 0.0004920
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026074, 0.0026140
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011898, 0.0011868
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005047, 0.0005059
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032795, 0.0032877
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008344, 0.0008324
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021590, 0.0021536
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011354, 0.0011325
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013132, 0.0013165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008385, upper bound: 0.0008080
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007948, upper bound: 0.0008545
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019746, 0.0019929
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004920, 0.0004966
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026316, 0.0026074
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011868, 0.0011978
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005093, 0.0005047
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0033098, 0.0032795
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008324, 0.0008401
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021536, 0.0021735
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011325, 0.0011430
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013254, 0.0013132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008545, upper bound: 0.0007948
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008080, upper bound: 0.0008385
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019818, 0.0019724
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004938, 0.0004915
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026045, 0.0026169
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011911, 0.0011855
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005041, 0.0005065
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032758, 0.0032913
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008354, 0.0008314
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021614, 0.0021512
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011366, 0.0011313
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013118, 0.0013180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008545, upper bound: 0.0007948
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008080, upper bound: 0.0008385
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019743, 0.0019919
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004919, 0.0004963
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026303, 0.0026070
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011866, 0.0011972
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005091, 0.0005046
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0033083, 0.0032789
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008322, 0.0008397
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021532, 0.0021725
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011324, 0.0011425
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013248, 0.0013130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008546, upper bound: 0.0007925
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008081, upper bound: 0.0008375
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019814, 0.0019721
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004937, 0.0004914
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026041, 0.0026165
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011909, 0.0011853
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005040, 0.0005064
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032753, 0.0032908
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008352, 0.0008313
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021610, 0.0021509
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011365, 0.0011311
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013116, 0.0013178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008546, upper bound: 0.0007925
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008081, upper bound: 0.0008375
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019721, 0.0019954
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004914, 0.0004972
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026348, 0.0026041
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011853, 0.0011993
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005100, 0.0005040
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0033139, 0.0032753
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008313, 0.0008411
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021508, 0.0021762
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011311, 0.0011445
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013270, 0.0013116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008376, upper bound: 0.0008071
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007931, upper bound: 0.0008545
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019792, 0.0019747
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004932, 0.0004920
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026075, 0.0026136
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011896, 0.0011868
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005047, 0.0005059
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032796, 0.0032872
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008343, 0.0008324
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021586, 0.0021536
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011352, 0.0011326
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013133, 0.0013163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008376, upper bound: 0.0008071
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007931, upper bound: 0.0008545
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019715, 0.0019945
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004913, 0.0004970
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026338, 0.0026034
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011849, 0.0011988
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005098, 0.0005039
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0033126, 0.0032744
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008311, 0.0008408
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021502, 0.0021753
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011308, 0.0011440
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013265, 0.0013112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008387, upper bound: 0.0008071
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007950, upper bound: 0.0008544
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9937361, 0.9965355, 0.9937361, 0.9965355, -0.0019787, 0.0019744
1: -0.0028247, -0.0021272, -0.0028247, -0.0021272, -0.0004930, 0.0004920
2: 0.0012192, 0.0049156, 0.0012192, 0.0049156, -0.0026071, 0.0026128
3: -0.0035105, -0.0018280, -0.0035105, -0.0018280, -0.0011892, 0.0011866
4: 0.0007639, 0.0014793, 0.0007639, 0.0014793, -0.0005046, 0.0005057
5: 0.0004928, 0.0051420, 0.0004928, 0.0051420, -0.0032791, 0.0032862
6: 0.0002357, 0.0014157, 0.0002357, 0.0014157, -0.0008341, 0.0008323
7: -0.0025277, 0.0005253, -0.0025277, 0.0005253, -0.0021580, 0.0021533
8: -0.0008934, 0.0007121, -0.0008934, 0.0007121, -0.0011349, 0.0011324
9: -0.0026896, -0.0008279, -0.0026896, -0.0008279, -0.0013131, 0.0013160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008387, upper bound: 0.0008071
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007950, upper bound: 0.0008544
time: 0.61 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.91 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008544, upper bound: 0.0007950
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008071, upper bound: 0.0008387
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008544, upper bound: 0.0007950
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008071, upper bound: 0.0008387
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008545, upper bound: 0.0007931
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008071, upper bound: 0.0008376
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008545, upper bound: 0.0007931
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008071, upper bound: 0.0008376
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008375, upper bound: 0.0008080
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0007925, upper bound: 0.0008546
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008375, upper bound: 0.0008080
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0007925, upper bound: 0.0008546
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008385, upper bound: 0.0008080
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0007948, upper bound: 0.0008545
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008385, upper bound: 0.0008080
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0007948, upper bound: 0.0008545
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008545, upper bound: 0.0007948
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008080, upper bound: 0.0008385
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008545, upper bound: 0.0007948
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008080, upper bound: 0.0008385
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008546, upper bound: 0.0007925
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008081, upper bound: 0.0008375
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008546, upper bound: 0.0007925
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008081, upper bound: 0.0008375
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008376, upper bound: 0.0008071
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0007931, upper bound: 0.0008545
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008376, upper bound: 0.0008071
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0007931, upper bound: 0.0008545
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008387, upper bound: 0.0008071
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0007950, upper bound: 0.0008544
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0008387, upper bound: 0.0008071
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.0007950, upper bound: 0.0008544

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.06 + 87.45 = 90.51 seconds
