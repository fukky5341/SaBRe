## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00390744


## IAR start

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
execution time: IAR + RelationalAnalysis = 1.72 + 3.69 = 5.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0043416, upper bound: 0.0043415

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040724, upper bound: 0.0040724
time: 2.34 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040724, upper bound: 0.0040724
time: 2.50 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.01 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.01
Output dim: 6, lower bound: -0.0040724, upper bound: 0.0040724
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.01
Output dim: 6, lower bound: -0.0040724, upper bound: 0.0040724

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683
time: 3.03 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683
time: 2.36 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 6.37 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.37
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.37
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.37
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.37
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0040683

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039899
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039893, upper bound: 0.0040526
time: 2.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039892
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039899, upper bound: 0.0040526
time: 2.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039899
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039893, upper bound: 0.0040526
time: 2.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039892
time: 2.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039899, upper bound: 0.0040526
time: 2.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 6.81 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.81
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039899
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.81
Output dim: 6, lower bound: -0.0039893, upper bound: 0.0040526
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.81
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039892
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.81
Output dim: 6, lower bound: -0.0039899, upper bound: 0.0040526
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.81
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039899
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.81
Output dim: 6, lower bound: -0.0039893, upper bound: 0.0040526
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.81
Output dim: 6, lower bound: -0.0040527, upper bound: 0.0039892
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.81
Output dim: 6, lower bound: -0.0039899, upper bound: 0.0040526

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040179, upper bound: 0.0039555
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0039554
time: 3.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039547, upper bound: 0.0040172
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0040178
time: 2.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040178, upper bound: 0.0039546
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040173, upper bound: 0.0039547
time: 2.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039555, upper bound: 0.0040173
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039555, upper bound: 0.0040178
time: 2.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040179, upper bound: 0.0039555
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040173, upper bound: 0.0039555
time: 2.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039547, upper bound: 0.0040173
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0040178
time: 2.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040179, upper bound: 0.0039548
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040173, upper bound: 0.0039547
time: 2.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039555, upper bound: 0.0040173
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0040179
time: 2.44 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 6.66 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0040179, upper bound: 0.0039555
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0039554
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0039547, upper bound: 0.0040172
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0040178
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0040178, upper bound: 0.0039546
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0040173, upper bound: 0.0039547
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0039555, upper bound: 0.0040173
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0039555, upper bound: 0.0040178
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0040179, upper bound: 0.0039555
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0040173, upper bound: 0.0039555
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0039547, upper bound: 0.0040173
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0040178
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0040179, upper bound: 0.0039548
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0040173, upper bound: 0.0039547
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0039555, upper bound: 0.0040173
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.66
Output dim: 6, lower bound: -0.0039548, upper bound: 0.0040179

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
time: 2.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037012
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037011
time: 2.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
time: 2.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037417
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037418
time: 1.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037418, upper bound: 0.0037010
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037418, upper bound: 0.0037010
time: 2.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037406, upper bound: 0.0037010
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037010
time: 2.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037419
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037419
time: 1.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037012
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037012
time: 2.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
time: 2.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
time: 2.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037417
time: 2.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037418
time: 2.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037418, upper bound: 0.0037010
time: 2.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037010
time: 2.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037406, upper bound: 0.0037010
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037406, upper bound: 0.0037010
time: 1.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037419
time: 2.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037418
time: 2.19 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 6.27 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037012
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037011
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037417
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037418
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037418, upper bound: 0.0037010
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037418, upper bound: 0.0037010
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037406, upper bound: 0.0037010
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037010
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037419
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037419
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037419, upper bound: 0.0037011
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037012
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037407, upper bound: 0.0037012
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037406
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037417
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037010, upper bound: 0.0037418
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037418, upper bound: 0.0037010
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037010
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037406, upper bound: 0.0037010
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037406, upper bound: 0.0037010
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037407
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037419
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 6, lower bound: -0.0037011, upper bound: 0.0037418

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 5.40 + 200.27 = 205.67 seconds
