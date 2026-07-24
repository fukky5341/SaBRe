## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0071008


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045)
1: (0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124672, 0.0124672)
2: (-0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994)
3: (-0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0023039, 0.0023039)
4: (-0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755)
5: (-0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050)
6: (-0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921)
7: (-0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030)
8: (-0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277)
9: (-0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.56 + 2.16 = 3.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0112707, upper bound: 0.0112707

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0110227, upper bound: 0.0110227
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0110227, upper bound: 0.0110227
time: 0.99 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.22
Output dim: 1, lower bound: -0.0110227, upper bound: 0.0110227
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.22
Output dim: 1, lower bound: -0.0110227, upper bound: 0.0110227

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124371, 0.0124385
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022924, 0.0022918
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107890, upper bound: 0.0107890
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107890, upper bound: 0.0107890
time: 1.21 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124385, 0.0124371
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022918, 0.0022924
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107890, upper bound: 0.0107890
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107890, upper bound: 0.0107890
time: 1.13 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 1, lower bound: -0.0107890, upper bound: 0.0107890
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 1, lower bound: -0.0107890, upper bound: 0.0107890
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 1, lower bound: -0.0107890, upper bound: 0.0107890
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 1, lower bound: -0.0107890, upper bound: 0.0107890

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124274, 0.0124270
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022886, 0.0022888
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089701
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089701
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124256, 0.0124288
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022894, 0.0022881
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089684
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089684
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124288, 0.0124256
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022881, 0.0022894
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089701
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089701
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124270, 0.0124274
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022888, 0.0022886
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089684
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089684
time: 0.66 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.87 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089701
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089701
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089684
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089684
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089701
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089701
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089684
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089684

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124236, 0.0124230
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022870, 0.0022873
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089624, upper bound: 0.0089701
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089634
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124234, 0.0124270
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022886, 0.0022872
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089624, upper bound: 0.0089701
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089634
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124216, 0.0124247
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022878, 0.0022864
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089634, upper bound: 0.0089684
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089624
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124216, 0.0124288
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022894, 0.0022864
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089634, upper bound: 0.0089684
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089624
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124250, 0.0124216
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022864, 0.0022879
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089624, upper bound: 0.0089701
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089634
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124247, 0.0124256
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022881, 0.0022878
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089624, upper bound: 0.0089701
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089634
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124227, 0.0124234
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022872, 0.0022869
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089634, upper bound: 0.0089684
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089624
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124230, 0.0124274
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022888, 0.0022870
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089634, upper bound: 0.0089684
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089624
time: 0.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089624, upper bound: 0.0089701
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089634
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089624, upper bound: 0.0089701
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089634
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089634, upper bound: 0.0089684
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089624
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089634, upper bound: 0.0089684
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089624
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089624, upper bound: 0.0089701
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089634
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089624, upper bound: 0.0089701
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089684, upper bound: 0.0089634
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089634, upper bound: 0.0089684
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089624
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089634, upper bound: 0.0089684
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.08
Output dim: 1, lower bound: -0.0089701, upper bound: 0.0089624

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124398, 0.0124393
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022929, 0.0022931
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124399, 0.0124402
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022932, 0.0022931
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124400, 0.0124434
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022945, 0.0022932
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124397, 0.0124442
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022949, 0.0022931
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124379, 0.0124411
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022936, 0.0022923
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124379, 0.0124421
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022941, 0.0022923
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124382, 0.0124451
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022952, 0.0022924
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124379, 0.0124462
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022957, 0.0022923
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124421, 0.0124379
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022923, 0.0022941
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124413, 0.0124382
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022924, 0.0022937
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124421, 0.0124420
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022939, 0.0022941
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124411, 0.0124422
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022940, 0.0022936
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124396, 0.0124398
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022931, 0.0022930
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124390, 0.0124400
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022932, 0.0022928
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124402, 0.0124438
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022947, 0.0022932
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124393, 0.0124440
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022948, 0.0022929
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058
time: 0.78 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084058, upper bound: 0.0084158
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084089, upper bound: 0.0084128
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084128, upper bound: 0.0084089
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.04
Output dim: 1, lower bound: -0.0084158, upper bound: 0.0084058

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123675, 0.0123845
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022704, 0.0022633
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123834, 0.0123670
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022631, 0.0022699
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123676, 0.0123817
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022692, 0.0022634
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123844, 0.0123678
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022635, 0.0022703
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123677, 0.0123885
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022720, 0.0022634
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123822, 0.0123710
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022647, 0.0022694
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123674, 0.0123857
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022708, 0.0022633
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123837, 0.0123719
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022651, 0.0022701
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123656, 0.0123862
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022711, 0.0022625
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123815, 0.0123687
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022638, 0.0022691
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123656, 0.0123831
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022698, 0.0022625
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123826, 0.0123698
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022643, 0.0022696
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123659, 0.0123902
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022727, 0.0022626
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123806, 0.0123728
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022654, 0.0022688
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123656, 0.0123872
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022714, 0.0022625
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123818, 0.0123738
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022659, 0.0022693
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123698, 0.0123818
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022693, 0.0022643
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123844, 0.0123656
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022625, 0.0022704
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123690, 0.0123806
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022688, 0.0022640
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123872, 0.0123659
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022626, 0.0022715
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123698, 0.0123859
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022709, 0.0022643
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123831, 0.0123696
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022641, 0.0022698
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123687, 0.0123846
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022704, 0.0022638
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123862, 0.0123699
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022643, 0.0022711
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123673, 0.0123837
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022701, 0.0022632
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123824, 0.0123674
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022633, 0.0022695
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123667, 0.0123822
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022694, 0.0022630
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123850, 0.0123677
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022634, 0.0022706
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123678, 0.0123878
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022717, 0.0022635
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123817, 0.0123715
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022649, 0.0022692
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123670, 0.0123862
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022710, 0.0022631
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123845, 0.0123717
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022650, 0.0022704
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
time: 0.76 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 6.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123782, 0.0124113
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022892, 0.0022754
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082515, upper bound: 0.0082398
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082093, upper bound: 0.0082698
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123907, 0.0123952
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022825, 0.0022806
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082621, upper bound: 0.0082283
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082214, upper bound: 0.0082606
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123942, 0.0123902
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022804, 0.0022820
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082515, upper bound: 0.0082395
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082093, upper bound: 0.0082698
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124083, 0.0123778
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022753, 0.0022879
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082621, upper bound: 0.0082283
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082214, upper bound: 0.0082606
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123784, 0.0124066
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022872, 0.0022755
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082529, upper bound: 0.0082319
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082110, upper bound: 0.0082679
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123895, 0.0123924
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022813, 0.0022801
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082645, upper bound: 0.0082213
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082241, upper bound: 0.0082583
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123952, 0.0123904
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022805, 0.0022825
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082529, upper bound: 0.0082319
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082110, upper bound: 0.0082679
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124104, 0.0123786
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022756, 0.0022888
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082645, upper bound: 0.0082213
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082242, upper bound: 0.0082583
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123785, 0.0124154
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022908, 0.0022755
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082515, upper bound: 0.0082398
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082093, upper bound: 0.0082698
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123902, 0.0123994
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022841, 0.0022804
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082621, upper bound: 0.0082283
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082214, upper bound: 0.0082606
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123930, 0.0123944
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022820, 0.0022815
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082515, upper bound: 0.0082395
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082093, upper bound: 0.0082698
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124070, 0.0123819
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022769, 0.0022874
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082621, upper bound: 0.0082283
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082214, upper bound: 0.0082606
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123782, 0.0124107
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022888, 0.0022754
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082529, upper bound: 0.0082319
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082110, upper bound: 0.0082679
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123900, 0.0123966
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022829, 0.0022803
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082645, upper bound: 0.0082213
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082241, upper bound: 0.0082583
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123945, 0.0123946
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022821, 0.0022822
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082529, upper bound: 0.0082319
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082110, upper bound: 0.0082679
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124101, 0.0123827
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022772, 0.0022887
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082645, upper bound: 0.0082213
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082242, upper bound: 0.0082583
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123764, 0.0124127
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022897, 0.0022747
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082583, upper bound: 0.0082242
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082213, upper bound: 0.0082645
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123883, 0.0123969
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022832, 0.0022796
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082679, upper bound: 0.0082110
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082319, upper bound: 0.0082529
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123923, 0.0123923
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022813, 0.0022813
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082583, upper bound: 0.0082241
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082213, upper bound: 0.0082645
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0124063, 0.0123795
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022760, 0.0022871
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082679, upper bound: 0.0082110
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082319, upper bound: 0.0082529
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123763, 0.0124084
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022879, 0.0022746
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082606, upper bound: 0.0082214
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082283, upper bound: 0.0082621
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0072033, 0.0057012, -0.0072033, 0.0057012, -0.0129045, 0.0129045
1: 0.9970881, 1.0113182, 0.9970881, 1.0113182, -0.0123874, 0.0123939
2: -0.0066773, 0.0063221, -0.0066773, 0.0063221, -0.0129994, 0.0129994
3: -0.0003022, 0.0025515, -0.0003022, 0.0025515, -0.0022819, 0.0022792
4: -0.0076269, 0.0017486, -0.0076269, 0.0017486, -0.0093755, 0.0093755
5: -0.0025010, 0.0091040, -0.0025010, 0.0091040, -0.0116050, 0.0116050
6: -0.0101468, 0.0020453, -0.0101468, 0.0020453, -0.0121921, 0.0121921
7: -0.0058313, 0.0004716, -0.0058313, 0.0004716, -0.0063030, 0.0063030
8: -0.0140483, -0.0016207, -0.0140483, -0.0016207, -0.0124277, 0.0124277
9: -0.0055716, 0.0079044, -0.0055716, 0.0079044, -0.0134760, 0.0134760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082698, upper bound: 0.0082093
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082395, upper bound: 0.0082515
time: 0.81 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 9.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082515, upper bound: 0.0082398
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082093, upper bound: 0.0082698
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082621, upper bound: 0.0082283
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082214, upper bound: 0.0082606
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082515, upper bound: 0.0082395
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082093, upper bound: 0.0082698
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082621, upper bound: 0.0082283
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082214, upper bound: 0.0082606
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082529, upper bound: 0.0082319
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082110, upper bound: 0.0082679
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082645, upper bound: 0.0082213
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082241, upper bound: 0.0082583
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082529, upper bound: 0.0082319
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082110, upper bound: 0.0082679
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082645, upper bound: 0.0082213
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082242, upper bound: 0.0082583
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082515, upper bound: 0.0082398
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082093, upper bound: 0.0082698
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082621, upper bound: 0.0082283
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082214, upper bound: 0.0082606
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082515, upper bound: 0.0082395
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082093, upper bound: 0.0082698
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082621, upper bound: 0.0082283
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082214, upper bound: 0.0082606
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082529, upper bound: 0.0082319
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082110, upper bound: 0.0082679
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082645, upper bound: 0.0082213
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082241, upper bound: 0.0082583
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082529, upper bound: 0.0082319
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082110, upper bound: 0.0082679
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082645, upper bound: 0.0082213
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082242, upper bound: 0.0082583
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082583, upper bound: 0.0082242
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082213, upper bound: 0.0082645
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082679, upper bound: 0.0082110
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082319, upper bound: 0.0082529
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082583, upper bound: 0.0082241
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082213, upper bound: 0.0082645
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082679, upper bound: 0.0082110
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082319, upper bound: 0.0082529
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082606, upper bound: 0.0082214
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082283, upper bound: 0.0082621
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082698, upper bound: 0.0082093
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.28
Output dim: 1, lower bound: -0.0082395, upper bound: 0.0082515
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083838, upper bound: 0.0084045
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083944, upper bound: 0.0083950
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083853, upper bound: 0.0084014
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083971, upper bound: 0.0083913
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083913, upper bound: 0.0083971
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0084014, upper bound: 0.0083853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0083950, upper bound: 0.0083944
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.28
Output dim: 1, lower bound: -0.0084045, upper bound: 0.0083838

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.72 + 597.00 = 600.72 seconds
