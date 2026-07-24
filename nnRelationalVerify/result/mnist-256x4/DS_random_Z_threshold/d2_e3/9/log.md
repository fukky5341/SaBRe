## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.030823649999999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650)
1: (-0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866)
2: (-0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139)
3: (-0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888)
4: (-0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017)
5: (0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867)
6: (-0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650)
7: (-0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367)
8: (-0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854)
9: (-0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.77 + 2.10 = 2.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.03 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.56 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.56
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.56
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 231

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 162

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307366, upper bound: 0.0307366
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307366, upper bound: 0.0307366
time: 1.71 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 231

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.23 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.99 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.99
Output dim: 5, lower bound: -0.0307366, upper bound: 0.0307366
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.99
Output dim: 5, lower bound: -0.0307366, upper bound: 0.0307366
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305993, upper bound: 0.0305993
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305993, upper bound: 0.0305993
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.22 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.44 seconds
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.44
Output dim: 5, lower bound: -0.0305993, upper bound: 0.0305993
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.44
Output dim: 5, lower bound: -0.0305993, upper bound: 0.0305993
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.51 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.19 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306989, upper bound: 0.0306989
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306989, upper bound: 0.0306989
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863
time: 0.80 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.33 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.33
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.33
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.33
Output dim: 5, lower bound: -0.0306989, upper bound: 0.0306989
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.33
Output dim: 5, lower bound: -0.0306989, upper bound: 0.0306989
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.33
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.33
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0287904, upper bound: 0.0287905
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0287904, upper bound: 0.0287905
time: 1.00 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.71 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.71
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.71
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.71
Output dim: 5, lower bound: -0.0287904, upper bound: 0.0287905
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.71
Output dim: 5, lower bound: -0.0287904, upper bound: 0.0287905

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.32 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.50 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 87

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 162

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306419, upper bound: 0.0306419
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306419, upper bound: 0.0306419
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.16 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.09 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.09
Output dim: 5, lower bound: -0.0306419, upper bound: 0.0306419
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.09
Output dim: 5, lower bound: -0.0306419, upper bound: 0.0306419
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0303783, upper bound: 0.0303783
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0303783, upper bound: 0.0303783
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 162

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0303235, upper bound: 0.0303235
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0303235, upper bound: 0.0303235
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310136, upper bound: 0.0310136
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310136, upper bound: 0.0310136
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0289966, upper bound: 0.0289966
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0289966, upper bound: 0.0289966
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0293794, upper bound: 0.0293794
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0293794, upper bound: 0.0293794
time: 1.49 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 3.71 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.71
Output dim: 5, lower bound: -0.0303783, upper bound: 0.0303783
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.71
Output dim: 5, lower bound: -0.0303783, upper bound: 0.0303783
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.71
Output dim: 5, lower bound: -0.0303235, upper bound: 0.0303235
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.71
Output dim: 5, lower bound: -0.0303235, upper bound: 0.0303235
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.71
Output dim: 5, lower bound: -0.0310136, upper bound: 0.0310136
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.71
Output dim: 5, lower bound: -0.0310136, upper bound: 0.0310136
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.71
Output dim: 5, lower bound: -0.0289966, upper bound: 0.0289966
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.71
Output dim: 5, lower bound: -0.0289966, upper bound: 0.0289966
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.71
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.71
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.71
Output dim: 5, lower bound: -0.0293794, upper bound: 0.0293794
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.71
Output dim: 5, lower bound: -0.0293794, upper bound: 0.0293794

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 162

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299492, upper bound: 0.0299492
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299492, upper bound: 0.0299492
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307157, upper bound: 0.0307157
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307157, upper bound: 0.0307157
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 87

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863
time: 0.88 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 2.49 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.49
Output dim: 5, lower bound: -0.0299492, upper bound: 0.0299492
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.49
Output dim: 5, lower bound: -0.0299492, upper bound: 0.0299492
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.49
Output dim: 5, lower bound: -0.0307157, upper bound: 0.0307157
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.49
Output dim: 5, lower bound: -0.0307157, upper bound: 0.0307157
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.49
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.49
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.49
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.49
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 162

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863
time: 0.91 seconds

## Summary of splitting (split count: 10)
- Time for DS candidates: 2.55 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 11, time: 2.55
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 11, time: 2.55
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 2.55
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 2.55
Output dim: 5, lower bound: -0.0260863, upper bound: 0.0260863

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0295958, upper bound: 0.0295958
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0295958, upper bound: 0.0295958
time: 1.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.98 seconds

## Summary of splitting (split count: 11)
- Time for DS candidates: 4.35 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 12, time: 4.35
Output dim: 5, lower bound: -0.0295958, upper bound: 0.0295958
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 12, time: 4.35
Output dim: 5, lower bound: -0.0295958, upper bound: 0.0295958
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 12, time: 4.35
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 12, time: 4.35
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0296642, upper bound: 0.0296642
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0296642, upper bound: 0.0296642
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650
1: -0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866
2: -0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139
3: -0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888
4: -0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017
5: 0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867
6: -0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650
7: -0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367
8: -0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854
9: -0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0293129, upper bound: 0.0293129
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0293129, upper bound: 0.0293129
time: 1.58 seconds

## Summary of splitting (split count: 12)
- Time for DS candidates: 5.75 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 13, time: 5.75
Output dim: 5, lower bound: -0.0296642, upper bound: 0.0296642
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 13, time: 5.75
Output dim: 5, lower bound: -0.0296642, upper bound: 0.0296642
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 13, time: 5.75
Output dim: 5, lower bound: -0.0293129, upper bound: 0.0293129
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 13, time: 5.75
Output dim: 5, lower bound: -0.0293129, upper bound: 0.0293129

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.87 + 123.82 = 126.70 seconds
