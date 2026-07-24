## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00390744


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263)
1: (-0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463)
2: (-0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734)
3: (-0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484)
4: (-0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258)
5: (-0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367)
6: (0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471)
7: (-0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576)
8: (-0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675)
9: (-0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 3.79 = 5.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0043416, upper bound: 0.0043415

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040724, upper bound: 0.0040724
time: 2.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040724, upper bound: 0.0040724
time: 2.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 5.17 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 5.17
Output dim: 6, lower bound: -0.0040724, upper bound: 0.0040724
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 5.17
Output dim: 6, lower bound: -0.0040724, upper bound: 0.0040724

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683
time: 3.14 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683
time: 2.44 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 6.05 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.05
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.05
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.05
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.05
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039899
time: 2.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039893, upper bound: 0.0040526
time: 2.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039892
time: 2.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039899, upper bound: 0.0040526
time: 2.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039899
time: 2.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039893, upper bound: 0.0040526
time: 2.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039892
time: 2.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039899, upper bound: 0.0040526
time: 2.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.24
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039899
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.24
Output dim: 6, lower bound: -0.0039893, upper bound: 0.0040526
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.24
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039892
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.24
Output dim: 6, lower bound: -0.0039899, upper bound: 0.0040526
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.24
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039899
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.24
Output dim: 6, lower bound: -0.0039893, upper bound: 0.0040526
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.24
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039892
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.24
Output dim: 6, lower bound: -0.0039899, upper bound: 0.0040526

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040179, upper bound: 0.0039555
time: 2.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0039554
time: 3.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039547, upper bound: 0.0040172
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0040178
time: 2.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040178, upper bound: 0.0039546
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040173, upper bound: 0.0039547
time: 2.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039555, upper bound: 0.0040173
time: 2.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039555, upper bound: 0.0040178
time: 2.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040179, upper bound: 0.0039555
time: 2.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040173, upper bound: 0.0039555
time: 2.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039547, upper bound: 0.0040173
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0040178
time: 2.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040179, upper bound: 0.0039548
time: 2.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040173, upper bound: 0.0039547
time: 2.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039555, upper bound: 0.0040173
time: 2.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0040179
time: 2.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0040179, upper bound: 0.0039555
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0039554
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0039547, upper bound: 0.0040172
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0040178
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0040178, upper bound: 0.0039546
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0040173, upper bound: 0.0039547
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0039555, upper bound: 0.0040173
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0039555, upper bound: 0.0040178
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0040179, upper bound: 0.0039555
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0040173, upper bound: 0.0039555
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0039547, upper bound: 0.0040173
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0040178
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0040179, upper bound: 0.0039548
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0040173, upper bound: 0.0039547
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0039555, upper bound: 0.0040173
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.34
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0040179

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
time: 2.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
time: 2.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037012
time: 2.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037011
time: 2.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
time: 2.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
time: 2.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037417
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037418
time: 1.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037418, upper bound: 0.0037010
time: 2.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037418, upper bound: 0.0037010
time: 2.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037406, upper bound: 0.0037010
time: 3.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037010
time: 2.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
time: 2.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
time: 1.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037419
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037419
time: 2.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
time: 2.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
time: 2.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037012
time: 2.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037012
time: 2.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
time: 2.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
time: 2.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037417
time: 2.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037418
time: 2.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037418, upper bound: 0.0037010
time: 2.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037010
time: 2.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037406, upper bound: 0.0037010
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037406, upper bound: 0.0037010
time: 2.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
time: 2.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
time: 2.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263
1: -0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463
2: -0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734
3: -0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484
4: -0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258
5: -0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367
6: 0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471
7: -0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576
8: -0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675
9: -0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037419
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037418
time: 2.27 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037012
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037011
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037417
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037418
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037418, upper bound: 0.0037010
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037418, upper bound: 0.0037010
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037406, upper bound: 0.0037010
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037010
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037419
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037419
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037012
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037012
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037417
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037418
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037418, upper bound: 0.0037010
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037010
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037406, upper bound: 0.0037010
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037406, upper bound: 0.0037010
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037419
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.06
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037418

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 5.11 + 191.26 = 196.37 seconds
