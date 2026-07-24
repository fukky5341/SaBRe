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
execution time: IAR + RelationalAnalysis = 0.75 + 3.55 = 4.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0043416, upper bound: 0.0043415

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0042548, upper bound: 0.0042548
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0042548, upper bound: 0.0042548
time: 1.97 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.87 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.87
Output dim: 6, lower bound: -0.0042548, upper bound: 0.0042548
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.87
Output dim: 6, lower bound: -0.0042548, upper bound: 0.0042548

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036680, upper bound: 0.0036680
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036680, upper bound: 0.0036680
time: 1.73 seconds

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0041767, upper bound: 0.0041462
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0041465, upper bound: 0.0041767
time: 1.80 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.51 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 4.51
Output dim: 6, lower bound: -0.0036680, upper bound: 0.0036680
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 4.51
Output dim: 6, lower bound: -0.0036680, upper bound: 0.0036680
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.51
Output dim: 6, lower bound: -0.0041767, upper bound: 0.0041462
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.51
Output dim: 6, lower bound: -0.0041465, upper bound: 0.0041767

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039696, upper bound: 0.0039431
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039696, upper bound: 0.0039431
time: 1.87 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039113, upper bound: 0.0039440
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039113, upper bound: 0.0039441
time: 2.16 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.11 seconds
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.11
Output dim: 6, lower bound: -0.0039696, upper bound: 0.0039431
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.11
Output dim: 6, lower bound: -0.0039696, upper bound: 0.0039431
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.11
Output dim: 6, lower bound: -0.0039113, upper bound: 0.0039440
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.11
Output dim: 6, lower bound: -0.0039113, upper bound: 0.0039441

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039447, upper bound: 0.0038985
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039164, upper bound: 0.0039162
time: 1.94 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037046, upper bound: 0.0036712
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037036, upper bound: 0.0036744
time: 2.22 seconds

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 75

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038871, upper bound: 0.0039146
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038837, upper bound: 0.0039202
time: 1.96 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037320, upper bound: 0.0037514
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037143, upper bound: 0.0037689
time: 1.80 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.10 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 6, lower bound: -0.0039447, upper bound: 0.0038985
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 6, lower bound: -0.0039164, upper bound: 0.0039162
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.10
Output dim: 6, lower bound: -0.0037046, upper bound: 0.0036712
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.10
Output dim: 6, lower bound: -0.0037036, upper bound: 0.0036744
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 6, lower bound: -0.0038871, upper bound: 0.0039146
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.10
Output dim: 6, lower bound: -0.0038837, upper bound: 0.0039202
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.10
Output dim: 6, lower bound: -0.0037320, upper bound: 0.0037514
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.10
Output dim: 6, lower bound: -0.0037143, upper bound: 0.0037689

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038690, upper bound: 0.0038218
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038690, upper bound: 0.0038218
time: 2.11 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034468, upper bound: 0.0034303
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034468, upper bound: 0.0034303
time: 1.90 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 237

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038736, upper bound: 0.0038919
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038620, upper bound: 0.0039004
time: 2.22 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 129

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034218, upper bound: 0.0034553
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034218, upper bound: 0.0034554
time: 2.28 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.30 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.30
Output dim: 6, lower bound: -0.0038690, upper bound: 0.0038218
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.30
Output dim: 6, lower bound: -0.0038690, upper bound: 0.0038218
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.30
Output dim: 6, lower bound: -0.0034468, upper bound: 0.0034303
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.30
Output dim: 6, lower bound: -0.0034468, upper bound: 0.0034303
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.30
Output dim: 6, lower bound: -0.0038736, upper bound: 0.0038919
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.30
Output dim: 6, lower bound: -0.0038620, upper bound: 0.0039004
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.30
Output dim: 6, lower bound: -0.0034218, upper bound: 0.0034553
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.30
Output dim: 6, lower bound: -0.0034218, upper bound: 0.0034554

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.30 + 65.48 = 69.78 seconds
