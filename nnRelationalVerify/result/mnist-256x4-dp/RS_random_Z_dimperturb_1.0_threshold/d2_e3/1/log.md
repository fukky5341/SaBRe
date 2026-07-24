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
execution time: IAR + RelationalAnalysis = 1.31 + 3.78 = 5.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0043416, upper bound: 0.0043415

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0042072, upper bound: 0.0042071
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0042072, upper bound: 0.0042071
time: 2.43 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 4.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 4.66
Output dim: 6, lower bound: -0.0042072, upper bound: 0.0042071
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 4.66
Output dim: 6, lower bound: -0.0042072, upper bound: 0.0042071

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040680, upper bound: 0.0040467
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040467, upper bound: 0.0040680
time: 2.49 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0039038, upper bound: 0.0039037
time: 2.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0039037, upper bound: 0.0039038
time: 2.25 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.91 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.91
Output dim: 6, lower bound: -0.0040680, upper bound: 0.0040467
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.91
Output dim: 6, lower bound: -0.0040467, upper bound: 0.0040680
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 5.91
Output dim: 6, lower bound: -0.0039038, upper bound: 0.0039037
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 5.91
Output dim: 6, lower bound: -0.0039037, upper bound: 0.0039038

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033993, upper bound: 0.0033921
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033993, upper bound: 0.0033921
time: 1.51 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040168, upper bound: 0.0040378
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040156, upper bound: 0.0040383
time: 1.97 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.08 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 5.08
Output dim: 6, lower bound: -0.0033993, upper bound: 0.0033921
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 5.08
Output dim: 6, lower bound: -0.0033993, upper bound: 0.0033921
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.08
Output dim: 6, lower bound: -0.0040168, upper bound: 0.0040378
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.08
Output dim: 6, lower bound: -0.0040156, upper bound: 0.0040383

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037719, upper bound: 0.0037847
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037719, upper bound: 0.0037847
time: 2.00 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040055, upper bound: 0.0040014
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039861, upper bound: 0.0040287
time: 1.88 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.12 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.12
Output dim: 6, lower bound: -0.0037719, upper bound: 0.0037847
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.12
Output dim: 6, lower bound: -0.0037719, upper bound: 0.0037847
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.12
Output dim: 6, lower bound: -0.0040055, upper bound: 0.0040014
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.12
Output dim: 6, lower bound: -0.0039861, upper bound: 0.0040287

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038145, upper bound: 0.0037990
time: 2.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038145, upper bound: 0.0037990
time: 2.32 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035267, upper bound: 0.0035609
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035267, upper bound: 0.0035609
time: 2.06 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 5.36 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.36
Output dim: 6, lower bound: -0.0038145, upper bound: 0.0037990
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.36
Output dim: 6, lower bound: -0.0038145, upper bound: 0.0037990
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.36
Output dim: 6, lower bound: -0.0035267, upper bound: 0.0035609
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.36
Output dim: 6, lower bound: -0.0035267, upper bound: 0.0035609

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 5.09 + 47.08 = 52.16 seconds
