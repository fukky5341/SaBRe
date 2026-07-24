## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01341639


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747)
1: (-0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919)
2: (-0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677)
3: (-0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820)
4: (-0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227)
5: (-0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0331142, 0.0331142)
6: (-0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399)
7: (-0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872)
8: (-0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206)
9: (-0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.20 + 3.36 = 5.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0149071, upper bound: 0.0149071

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0148995, upper bound: 0.0142320
time: 2.28 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0142320, upper bound: 0.0148995
time: 2.05 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.54 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.54
Output dim: 0, lower bound: -0.0148995, upper bound: 0.0142320
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.54
Output dim: 0, lower bound: -0.0142320, upper bound: 0.0148995

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0313846, 0.0317231
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0148306, upper bound: 0.0141617
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0148306, upper bound: 0.0141616
time: 2.02 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0317231, 0.0313846
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0141616, upper bound: 0.0148306
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0141618, upper bound: 0.0148306
time: 1.98 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 6.61 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.61
Output dim: 0, lower bound: -0.0148306, upper bound: 0.0141617
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.61
Output dim: 0, lower bound: -0.0148306, upper bound: 0.0141616
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.61
Output dim: 0, lower bound: -0.0141616, upper bound: 0.0148306
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.61
Output dim: 0, lower bound: -0.0141618, upper bound: 0.0148306

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0313244, 0.0316600
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125859, upper bound: 0.0123066
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125859, upper bound: 0.0123066
time: 1.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0313215, 0.0316532
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125859, upper bound: 0.0123066
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125859, upper bound: 0.0123066
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0316532, 0.0313215
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123066, upper bound: 0.0125859
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123066, upper bound: 0.0125859
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0316600, 0.0313244
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123066, upper bound: 0.0125859
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123066, upper bound: 0.0125859
time: 1.49 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.14 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.14
Output dim: 0, lower bound: -0.0125859, upper bound: 0.0123066
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.14
Output dim: 0, lower bound: -0.0125859, upper bound: 0.0123066
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.14
Output dim: 0, lower bound: -0.0125859, upper bound: 0.0123066
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.14
Output dim: 0, lower bound: -0.0125859, upper bound: 0.0123066
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.14
Output dim: 0, lower bound: -0.0123066, upper bound: 0.0125859
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.14
Output dim: 0, lower bound: -0.0123066, upper bound: 0.0125859
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.14
Output dim: 0, lower bound: -0.0123066, upper bound: 0.0125859
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.14
Output dim: 0, lower bound: -0.0123066, upper bound: 0.0125859

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 5.56 + 38.10 = 43.67 seconds
