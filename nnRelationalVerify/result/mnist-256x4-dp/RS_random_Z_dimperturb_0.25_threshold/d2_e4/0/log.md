## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00070371


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515)
1: (-0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144)
2: (0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701)
3: (-0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091)
4: (-0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109)
5: (-0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833)
6: (0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342)
7: (-0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0066834, 0.0066834)
8: (-0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333)
9: (-0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.48 + 2.10 = 3.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0008699, upper bound: 0.0008699

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 60

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008681, upper bound: 0.0008671
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008672, upper bound: 0.0008681
time: 2.21 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.40 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.40
Output dim: 6, lower bound: -0.0008681, upper bound: 0.0008671
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.40
Output dim: 6, lower bound: -0.0008672, upper bound: 0.0008681

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0066677, 0.0066626
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008527, upper bound: 0.0008514
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008524, upper bound: 0.0008517
time: 1.14 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0066626, 0.0066677
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006726, upper bound: 0.0006731
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006726, upper bound: 0.0006731
time: 0.86 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 6, lower bound: -0.0008527, upper bound: 0.0008514
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 6, lower bound: -0.0008524, upper bound: 0.0008517
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.04
Output dim: 6, lower bound: -0.0006726, upper bound: 0.0006731
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.04
Output dim: 6, lower bound: -0.0006726, upper bound: 0.0006731

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0065125, 0.0065022
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008271, upper bound: 0.0008262
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008273, upper bound: 0.0008258
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0065073, 0.0065077
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008269, upper bound: 0.0008262
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008273, upper bound: 0.0008258
time: 1.15 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.61 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 6, lower bound: -0.0008271, upper bound: 0.0008262
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 6, lower bound: -0.0008273, upper bound: 0.0008258
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 6, lower bound: -0.0008269, upper bound: 0.0008262
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 6, lower bound: -0.0008273, upper bound: 0.0008258

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059375, 0.0059105
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008192, upper bound: 0.0008197
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008185, upper bound: 0.0008188
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059087, 0.0059272
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007852, upper bound: 0.0007831
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007851, upper bound: 0.0007835
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059323, 0.0059021
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008252, upper bound: 0.0008130
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008118, upper bound: 0.0008245
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059186, 0.0059327
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008261, upper bound: 0.0008245
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008255, upper bound: 0.0008247
time: 1.14 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 6, lower bound: -0.0008192, upper bound: 0.0008197
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 6, lower bound: -0.0008185, upper bound: 0.0008188
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 6, lower bound: -0.0007852, upper bound: 0.0007831
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 6, lower bound: -0.0007851, upper bound: 0.0007835
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 6, lower bound: -0.0008252, upper bound: 0.0008130
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 6, lower bound: -0.0008118, upper bound: 0.0008245
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 6, lower bound: -0.0008261, upper bound: 0.0008245
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 6, lower bound: -0.0008255, upper bound: 0.0008247

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057531, 0.0057054
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008009, upper bound: 0.0008011
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007991, upper bound: 0.0008016
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057406, 0.0057261
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007689, upper bound: 0.0007662
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007689, upper bound: 0.0007662
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059080, 0.0059255
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005592, upper bound: 0.0005589
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005592, upper bound: 0.0005589
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059071, 0.0059272
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007147, upper bound: 0.0007129
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007147, upper bound: 0.0007129
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059967, 0.0059375
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006996, upper bound: 0.0006884
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006996, upper bound: 0.0006884
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059677, 0.0059696
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007683, upper bound: 0.0007823
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007697, upper bound: 0.0007806
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059204, 0.0059372
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006337, upper bound: 0.0006327
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006337, upper bound: 0.0006327
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059184, 0.0059345
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006339, upper bound: 0.0006326
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006339, upper bound: 0.0006326
time: 1.00 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0008009, upper bound: 0.0008011
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0007991, upper bound: 0.0008016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0007689, upper bound: 0.0007662
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0007689, upper bound: 0.0007662
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0005592, upper bound: 0.0005589
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0005592, upper bound: 0.0005589
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0007147, upper bound: 0.0007129
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0007147, upper bound: 0.0007129
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0006996, upper bound: 0.0006884
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0006996, upper bound: 0.0006884
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0007683, upper bound: 0.0007823
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0007697, upper bound: 0.0007806
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0006337, upper bound: 0.0006327
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0006337, upper bound: 0.0006327
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0006339, upper bound: 0.0006326
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.40
Output dim: 6, lower bound: -0.0006339, upper bound: 0.0006326

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056881, 0.0056242
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006756, upper bound: 0.0006755
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006756, upper bound: 0.0006755
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056720, 0.0056356
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007970, upper bound: 0.0007987
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007964, upper bound: 0.0007993
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057213, 0.0057102
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007663, upper bound: 0.0007634
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007659, upper bound: 0.0007635
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057247, 0.0057261
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007495, upper bound: 0.0007438
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007458, upper bound: 0.0007468
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0058876, 0.0059109
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007085, upper bound: 0.0007078
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007097, upper bound: 0.0007067
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0058908, 0.0059272
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007111, upper bound: 0.0007098
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007116, upper bound: 0.0007095
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0058259, 0.0058591
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007657, upper bound: 0.0007792
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007649, upper bound: 0.0007797
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059677, 0.0058278
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007251, upper bound: 0.0007352
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007251, upper bound: 0.0007351
time: 1.02 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0006756, upper bound: 0.0006755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0006756, upper bound: 0.0006755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0007970, upper bound: 0.0007987
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0007964, upper bound: 0.0007993
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0007663, upper bound: 0.0007634
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0007659, upper bound: 0.0007635
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0007495, upper bound: 0.0007438
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0007458, upper bound: 0.0007468
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0007085, upper bound: 0.0007078
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0007097, upper bound: 0.0007067
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0007111, upper bound: 0.0007098
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0007116, upper bound: 0.0007095
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0007657, upper bound: 0.0007792
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0007649, upper bound: 0.0007797
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0007251, upper bound: 0.0007352
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 6, lower bound: -0.0007251, upper bound: 0.0007351

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056166, 0.0055775
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007763, upper bound: 0.0007781
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007764, upper bound: 0.0007782
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056139, 0.0055821
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007849, upper bound: 0.0007900
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007851, upper bound: 0.0007930
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056712, 0.0056550
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007243, upper bound: 0.0007214
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007245, upper bound: 0.0007204
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056661, 0.0056593
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007600, upper bound: 0.0007576
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007600, upper bound: 0.0007576
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057026, 0.0056730
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007484, upper bound: 0.0007423
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007479, upper bound: 0.0007427
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056716, 0.0057015
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007241, upper bound: 0.0007236
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007219, upper bound: 0.0007241
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057035, 0.0057006
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006962, upper bound: 0.0006941
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006945, upper bound: 0.0006951
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056879, 0.0057269
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0007031, upper bound: 0.0007005
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0007034, upper bound: 0.0007003
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0058384, 0.0058708
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007099, upper bound: 0.0006987
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006986, upper bound: 0.0007086
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0058348, 0.0058706
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006997, upper bound: 0.0006957
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006987, upper bound: 0.0006967
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057662, 0.0058000
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007467, upper bound: 0.0007602
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007467, upper bound: 0.0007602
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057669, 0.0058020
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007632, upper bound: 0.0007784
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007639, upper bound: 0.0007787
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059672, 0.0058262
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006326, upper bound: 0.0006420
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006326, upper bound: 0.0006420
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059662, 0.0058278
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007224, upper bound: 0.0007319
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007215, upper bound: 0.0007323
time: 1.15 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007763, upper bound: 0.0007781
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007764, upper bound: 0.0007782
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007849, upper bound: 0.0007900
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007851, upper bound: 0.0007930
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007243, upper bound: 0.0007214
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007245, upper bound: 0.0007204
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007600, upper bound: 0.0007576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007600, upper bound: 0.0007576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007484, upper bound: 0.0007423
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007479, upper bound: 0.0007427
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007241, upper bound: 0.0007236
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007219, upper bound: 0.0007241
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0006962, upper bound: 0.0006941
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0006945, upper bound: 0.0006951
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007031, upper bound: 0.0007005
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007034, upper bound: 0.0007003
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007099, upper bound: 0.0006987
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0006986, upper bound: 0.0007086
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0006997, upper bound: 0.0006957
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0006987, upper bound: 0.0006967
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007467, upper bound: 0.0007602
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007467, upper bound: 0.0007602
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007632, upper bound: 0.0007784
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007639, upper bound: 0.0007787
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0006326, upper bound: 0.0006420
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0006326, upper bound: 0.0006420
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007224, upper bound: 0.0007319
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.65
Output dim: 6, lower bound: -0.0007215, upper bound: 0.0007323

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055105, 0.0054768
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007220, upper bound: 0.0007212
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007220, upper bound: 0.0007212
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055159, 0.0054750
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006567, upper bound: 0.0006595
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006567, upper bound: 0.0006596
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054829, 0.0054429
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007889, upper bound: 0.0007884
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007833, upper bound: 0.0007888
time: 1.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054746, 0.0054529
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007369, upper bound: 0.0007450
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007369, upper bound: 0.0007450
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055239, 0.0055575
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006807, upper bound: 0.0006882
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006916, upper bound: 0.0006797
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056712, 0.0055078
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006056, upper bound: 0.0005981
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006056, upper bound: 0.0005981
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055972, 0.0055877
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007434, upper bound: 0.0007411
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007427, upper bound: 0.0007413
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055923, 0.0055904
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007589, upper bound: 0.0007564
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007578, upper bound: 0.0007566
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056923, 0.0056617
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007424, upper bound: 0.0007364
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007424, upper bound: 0.0007358
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056909, 0.0056625
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007249, upper bound: 0.0007198
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007236, upper bound: 0.0007212
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056030, 0.0056208
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007175, upper bound: 0.0007172
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007176, upper bound: 0.0007171
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055916, 0.0056308
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007182, upper bound: 0.0007139
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007118, upper bound: 0.0007204
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059069, 0.0059084
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007088, upper bound: 0.0006976
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007086, upper bound: 0.0006974
time: 1.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0058757, 0.0059383
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006296, upper bound: 0.0006390
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006296, upper bound: 0.0006390
time: 1.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057097, 0.0057250
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006973, upper bound: 0.0007045
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006973, upper bound: 0.0007045
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056912, 0.0057384
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007135, upper bound: 0.0007240
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007106, upper bound: 0.0007256
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057653, 0.0058010
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007243, upper bound: 0.0007384
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007243, upper bound: 0.0007384
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057638, 0.0058005
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006984, upper bound: 0.0007072
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006984, upper bound: 0.0007072
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059114, 0.0057687
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007164, upper bound: 0.0007256
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007163, upper bound: 0.0007261
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059120, 0.0057712
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006891, upper bound: 0.0006968
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006874, upper bound: 0.0006994
time: 1.05 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 5.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007220, upper bound: 0.0007212
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007220, upper bound: 0.0007212
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0006567, upper bound: 0.0006595
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0006567, upper bound: 0.0006596
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007889, upper bound: 0.0007884
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007833, upper bound: 0.0007888
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007369, upper bound: 0.0007450
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007369, upper bound: 0.0007450
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0006807, upper bound: 0.0006882
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0006916, upper bound: 0.0006797
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0006056, upper bound: 0.0005981
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0006056, upper bound: 0.0005981
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007434, upper bound: 0.0007411
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007427, upper bound: 0.0007413
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007589, upper bound: 0.0007564
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007578, upper bound: 0.0007566
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007424, upper bound: 0.0007364
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007424, upper bound: 0.0007358
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007249, upper bound: 0.0007198
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007236, upper bound: 0.0007212
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007175, upper bound: 0.0007172
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007176, upper bound: 0.0007171
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007182, upper bound: 0.0007139
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007118, upper bound: 0.0007204
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007088, upper bound: 0.0006976
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007086, upper bound: 0.0006974
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0006296, upper bound: 0.0006390
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0006296, upper bound: 0.0006390
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0006973, upper bound: 0.0007045
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0006973, upper bound: 0.0007045
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007135, upper bound: 0.0007240
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007106, upper bound: 0.0007256
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007243, upper bound: 0.0007384
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007243, upper bound: 0.0007384
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0006984, upper bound: 0.0007072
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0006984, upper bound: 0.0007072
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007164, upper bound: 0.0007256
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0007163, upper bound: 0.0007261
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0006891, upper bound: 0.0006968
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.38
Output dim: 6, lower bound: -0.0006874, upper bound: 0.0006994

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054917, 0.0054611
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006426, upper bound: 0.0006388
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006426, upper bound: 0.0006388
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054949, 0.0054768
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007206, upper bound: 0.0007064
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007051, upper bound: 0.0007198
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054801, 0.0054397
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007876, upper bound: 0.0007771
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007762, upper bound: 0.0007871
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054818, 0.0054401
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007436, upper bound: 0.0007368
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007436, upper bound: 0.0007368
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054561, 0.0054372
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007353, upper bound: 0.0007309
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007233, upper bound: 0.0007436
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054589, 0.0054529
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006941, upper bound: 0.0007097
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0007024, upper bound: 0.0007013
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055314, 0.0055095
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006957, upper bound: 0.0006919
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006957, upper bound: 0.0006910
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055191, 0.0055223
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005099, upper bound: 0.0005089
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005099, upper bound: 0.0005089
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055861, 0.0055845
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007422, upper bound: 0.0007398
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007417, upper bound: 0.0007402
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055856, 0.0055842
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005156, upper bound: 0.0005115
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005156, upper bound: 0.0005115
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056374, 0.0056062
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006943, upper bound: 0.0006887
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006943, upper bound: 0.0006882
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056317, 0.0056072
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006982, upper bound: 0.0007023
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007086, upper bound: 0.0006933
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056238, 0.0055850
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006670, upper bound: 0.0006622
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006670, upper bound: 0.0006622
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056134, 0.0055978
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007206, upper bound: 0.0007181
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007206, upper bound: 0.0007174
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055505, 0.0055691
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007138, upper bound: 0.0007070
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007070, upper bound: 0.0007136
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055483, 0.0055690
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006913, upper bound: 0.0006918
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006913, upper bound: 0.0006918
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054796, 0.0055090
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006590, upper bound: 0.0006516
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006590, upper bound: 0.0006517
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054690, 0.0055199
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007108, upper bound: 0.0007192
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007102, upper bound: 0.0007194
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0059030, 0.0059039
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006607, upper bound: 0.0006539
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006659, upper bound: 0.0006507
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0058999, 0.0059043
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006606, upper bound: 0.0006538
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006658, upper bound: 0.0006507
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057092, 0.0057235
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006913, upper bound: 0.0006979
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006911, upper bound: 0.0006984
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057081, 0.0057250
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006962, upper bound: 0.0007027
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006962, upper bound: 0.0007034
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056507, 0.0056726
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006534, upper bound: 0.0006567
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006534, upper bound: 0.0006567
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056255, 0.0056976
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007093, upper bound: 0.0007243
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007097, upper bound: 0.0007245
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056551, 0.0057006
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007061, upper bound: 0.0007157
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007032, upper bound: 0.0007199
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056649, 0.0056943
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007171, upper bound: 0.0007321
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007180, upper bound: 0.0007321
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056469, 0.0057297
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006912, upper bound: 0.0007009
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006923, upper bound: 0.0007009
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057638, 0.0056836
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006590, upper bound: 0.0006736
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006646, upper bound: 0.0006669
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0058437, 0.0056982
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007119, upper bound: 0.0007168
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007082, upper bound: 0.0007211
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0058435, 0.0057039
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006695, upper bound: 0.0006780
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006695, upper bound: 0.0006780
time: 0.99 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 7.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006426, upper bound: 0.0006388
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006426, upper bound: 0.0006388
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007206, upper bound: 0.0007064
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007051, upper bound: 0.0007198
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007876, upper bound: 0.0007771
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007762, upper bound: 0.0007871
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007436, upper bound: 0.0007368
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007436, upper bound: 0.0007368
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007353, upper bound: 0.0007309
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007233, upper bound: 0.0007436
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006941, upper bound: 0.0007097
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007024, upper bound: 0.0007013
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006957, upper bound: 0.0006919
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006957, upper bound: 0.0006910
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0005099, upper bound: 0.0005089
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0005099, upper bound: 0.0005089
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007422, upper bound: 0.0007398
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007417, upper bound: 0.0007402
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0005156, upper bound: 0.0005115
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0005156, upper bound: 0.0005115
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006943, upper bound: 0.0006887
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006943, upper bound: 0.0006882
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006982, upper bound: 0.0007023
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007086, upper bound: 0.0006933
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006670, upper bound: 0.0006622
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006670, upper bound: 0.0006622
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007206, upper bound: 0.0007181
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007206, upper bound: 0.0007174
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007138, upper bound: 0.0007070
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007070, upper bound: 0.0007136
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006913, upper bound: 0.0006918
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006913, upper bound: 0.0006918
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006590, upper bound: 0.0006516
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006590, upper bound: 0.0006517
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007108, upper bound: 0.0007192
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007102, upper bound: 0.0007194
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006607, upper bound: 0.0006539
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006659, upper bound: 0.0006507
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006606, upper bound: 0.0006538
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006658, upper bound: 0.0006507
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006913, upper bound: 0.0006979
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006911, upper bound: 0.0006984
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006962, upper bound: 0.0007027
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006962, upper bound: 0.0007034
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006534, upper bound: 0.0006567
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006534, upper bound: 0.0006567
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007093, upper bound: 0.0007243
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007097, upper bound: 0.0007245
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007061, upper bound: 0.0007157
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007032, upper bound: 0.0007199
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007171, upper bound: 0.0007321
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007180, upper bound: 0.0007321
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006912, upper bound: 0.0007009
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006923, upper bound: 0.0007009
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006590, upper bound: 0.0006736
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006646, upper bound: 0.0006669
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007119, upper bound: 0.0007168
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0007082, upper bound: 0.0007211
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006695, upper bound: 0.0006780
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 7.45
Output dim: 6, lower bound: -0.0006695, upper bound: 0.0006780

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055611, 0.0055061
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006535, upper bound: 0.0006448
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006535, upper bound: 0.0006448
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055235, 0.0055330
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007042, upper bound: 0.0007182
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007042, upper bound: 0.0007188
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055384, 0.0054648
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006945, upper bound: 0.0006832
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006945, upper bound: 0.0006832
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055051, 0.0054898
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006524, upper bound: 0.0006596
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006524, upper bound: 0.0006596
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054628, 0.0054243
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005073, upper bound: 0.0005016
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005073, upper bound: 0.0005016
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054661, 0.0054401
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005073, upper bound: 0.0005016
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005073, upper bound: 0.0005016
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055267, 0.0054750
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006667, upper bound: 0.0006621
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006667, upper bound: 0.0006621
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054939, 0.0055016
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006972, upper bound: 0.0007133
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006958, upper bound: 0.0007158
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0052012, 0.0052806
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004570, upper bound: 0.0004714
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004570, upper bound: 0.0004714
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055229, 0.0055064
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007151, upper bound: 0.0007113
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007151, upper bound: 0.0007113
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055080, 0.0055169
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006709, upper bound: 0.0006694
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006709, upper bound: 0.0006694
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054457, 0.0053370
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006670, upper bound: 0.0006566
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006670, upper bound: 0.0006566
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055538, 0.0055329
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006590, upper bound: 0.0006559
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006590, upper bound: 0.0006559
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055486, 0.0055364
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006628, upper bound: 0.0006595
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006628, upper bound: 0.0006595
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054340, 0.0054427
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006527, upper bound: 0.0006453
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006527, upper bound: 0.0006453
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054239, 0.0054539
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006474, upper bound: 0.0006498
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006474, upper bound: 0.0006498
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054644, 0.0055144
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006312, upper bound: 0.0006359
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006312, upper bound: 0.0006359
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054648, 0.0055151
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006649, upper bound: 0.0006770
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006686, upper bound: 0.0006736
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056232, 0.0056928
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006429, upper bound: 0.0006518
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006429, upper bound: 0.0006518
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056236, 0.0056953
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004975, upper bound: 0.0004987
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004975, upper bound: 0.0004987
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0056147, 0.0056315
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006524, upper bound: 0.0006601
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006524, upper bound: 0.0006601
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055859, 0.0056544
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006975, upper bound: 0.0007144
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006978, upper bound: 0.0007144
time: 2.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054535, 0.0054666
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004800, upper bound: 0.0004805
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004800, upper bound: 0.0004805
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054271, 0.0054830
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006480, upper bound: 0.0006545
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006480, upper bound: 0.0006545
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057240, 0.0055808
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006651, upper bound: 0.0006670
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006651, upper bound: 0.0006670
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0057169, 0.0055870
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006336, upper bound: 0.0006414
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006336, upper bound: 0.0006414
time: 1.05 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 7.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006535, upper bound: 0.0006448
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006535, upper bound: 0.0006448
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0007042, upper bound: 0.0007182
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0007042, upper bound: 0.0007188
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006945, upper bound: 0.0006832
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006945, upper bound: 0.0006832
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006524, upper bound: 0.0006596
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006524, upper bound: 0.0006596
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0005073, upper bound: 0.0005016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0005073, upper bound: 0.0005016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0005073, upper bound: 0.0005016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0005073, upper bound: 0.0005016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006667, upper bound: 0.0006621
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006667, upper bound: 0.0006621
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006972, upper bound: 0.0007133
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006958, upper bound: 0.0007158
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0004570, upper bound: 0.0004714
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0004570, upper bound: 0.0004714
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0007151, upper bound: 0.0007113
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0007151, upper bound: 0.0007113
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006709, upper bound: 0.0006694
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006709, upper bound: 0.0006694
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006670, upper bound: 0.0006566
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006670, upper bound: 0.0006566
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006590, upper bound: 0.0006559
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006590, upper bound: 0.0006559
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006628, upper bound: 0.0006595
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006628, upper bound: 0.0006595
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006527, upper bound: 0.0006453
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006527, upper bound: 0.0006453
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006474, upper bound: 0.0006498
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006474, upper bound: 0.0006498
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006312, upper bound: 0.0006359
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006312, upper bound: 0.0006359
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006649, upper bound: 0.0006770
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006686, upper bound: 0.0006736
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006429, upper bound: 0.0006518
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006429, upper bound: 0.0006518
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0004975, upper bound: 0.0004987
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0004975, upper bound: 0.0004987
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006524, upper bound: 0.0006601
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006524, upper bound: 0.0006601
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006975, upper bound: 0.0007144
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006978, upper bound: 0.0007144
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0004800, upper bound: 0.0004805
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0004800, upper bound: 0.0004805
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006480, upper bound: 0.0006545
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006480, upper bound: 0.0006545
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006651, upper bound: 0.0006670
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006651, upper bound: 0.0006670
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006336, upper bound: 0.0006414
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.74
Output dim: 6, lower bound: -0.0006336, upper bound: 0.0006414

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055027, 0.0055084
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006427, upper bound: 0.0006550
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006427, upper bound: 0.0006550
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055017, 0.0055124
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007002, upper bound: 0.0007079
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006927, upper bound: 0.0007149
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054701, 0.0054454
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006177, upper bound: 0.0006320
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006177, upper bound: 0.0006320
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054378, 0.0054749
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006412, upper bound: 0.0006567
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006412, upper bound: 0.0006567
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054236, 0.0054042
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007137, upper bound: 0.0006976
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006985, upper bound: 0.0007099
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054207, 0.0053952
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0007137, upper bound: 0.0006976
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006985, upper bound: 0.0007099
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055210, 0.0055868
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006059, upper bound: 0.0006148
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006059, upper bound: 0.0006148
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0055214, 0.0055895
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006933, upper bound: 0.0007056
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006908, upper bound: 0.0007101
time: 1.23 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 7.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0006427, upper bound: 0.0006550
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0006427, upper bound: 0.0006550
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0007002, upper bound: 0.0007079
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0006927, upper bound: 0.0007149
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0006177, upper bound: 0.0006320
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0006177, upper bound: 0.0006320
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0006412, upper bound: 0.0006567
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0006412, upper bound: 0.0006567
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0007137, upper bound: 0.0006976
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0006985, upper bound: 0.0007099
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0007137, upper bound: 0.0006976
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0006985, upper bound: 0.0007099
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0006059, upper bound: 0.0006148
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0006059, upper bound: 0.0006148
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0006933, upper bound: 0.0007056
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 7.94
Output dim: 6, lower bound: -0.0006908, upper bound: 0.0007101

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0053904, 0.0053910
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006752, upper bound: 0.0006813
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006715, upper bound: 0.0006827
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0053804, 0.0054019
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006682, upper bound: 0.0006877
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006660, upper bound: 0.0006897
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054833, 0.0054266
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006335, upper bound: 0.0006202
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006335, upper bound: 0.0006202
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054459, 0.0054480
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006947, upper bound: 0.0007006
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006871, upper bound: 0.0007059
time: 2.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054768, 0.0054175
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006335, upper bound: 0.0006202
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006335, upper bound: 0.0006202
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054431, 0.0054440
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006377, upper bound: 0.0006447
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006377, upper bound: 0.0006447
time: 1.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054159, 0.0054741
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006868, upper bound: 0.0006991
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006869, upper bound: 0.0006991
time: 1.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0054059, 0.0054825
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006367, upper bound: 0.0006532
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006371, upper bound: 0.0006532
time: 1.19 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 4.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006752, upper bound: 0.0006813
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006715, upper bound: 0.0006827
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006682, upper bound: 0.0006877
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006660, upper bound: 0.0006897
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006335, upper bound: 0.0006202
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006335, upper bound: 0.0006202
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006947, upper bound: 0.0007006
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006871, upper bound: 0.0007059
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006335, upper bound: 0.0006202
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006335, upper bound: 0.0006202
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006377, upper bound: 0.0006447
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006377, upper bound: 0.0006447
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006868, upper bound: 0.0006991
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006869, upper bound: 0.0006991
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006367, upper bound: 0.0006532
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.04
Output dim: 6, lower bound: -0.0006371, upper bound: 0.0006532

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058266, 0.0064781, 0.0058266, 0.0064781, -0.0006515, 0.0006515
1: -0.0007848, 0.0006296, -0.0007848, 0.0006296, -0.0014144, 0.0014144
2: 0.0118499, 0.0222200, 0.0118499, 0.0222200, -0.0103701, 0.0103701
3: -0.0044801, -0.0035709, -0.0044801, -0.0035709, -0.0009091, 0.0009091
4: -0.0004003, 0.0040106, -0.0004003, 0.0040106, -0.0044109, 0.0044109
5: -0.0011184, -0.0003351, -0.0011184, -0.0003351, -0.0007833, 0.0007833
6: 0.9907701, 0.9923043, 0.9907701, 0.9923043, -0.0015342, 0.0015342
7: -0.0141705, -0.0061230, -0.0141705, -0.0061230, -0.0053261, 0.0053387
8: -0.0043632, -0.0009299, -0.0043632, -0.0009299, -0.0034333, 0.0034333
9: -0.0054731, -0.0003570, -0.0054731, -0.0003570, -0.0051161, 0.0051161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=244
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006089, upper bound: 0.0006196
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0006089, upper bound: 0.0006196
time: 1.13 seconds

## Summary of splitting (split count: 12)
- Time for RS candidates: 6.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 13, time: 6.10
Output dim: 6, lower bound: -0.0006089, upper bound: 0.0006196
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 13, time: 6.10
Output dim: 6, lower bound: -0.0006089, upper bound: 0.0006196

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.59 + 582.09 = 585.68 seconds
