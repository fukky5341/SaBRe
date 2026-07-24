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
execution time: IAR + RelationalAnalysis = 0.83 + 3.22 = 4.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0149071, upper bound: 0.0149071

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 231
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 231

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0148310, upper bound: 0.0146565
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146566, upper bound: 0.0148310
time: 1.87 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.86 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.86
Output dim: 0, lower bound: -0.0148310, upper bound: 0.0146565
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.86
Output dim: 0, lower bound: -0.0146566, upper bound: 0.0148310

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0328667, 0.0329331
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146392, upper bound: 0.0143266
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144969, upper bound: 0.0144644
time: 2.11 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0329331, 0.0328666
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146340, upper bound: 0.0148003
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146196, upper bound: 0.0148078
time: 1.95 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.99 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.99
Output dim: 0, lower bound: -0.0146392, upper bound: 0.0143266
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.99
Output dim: 0, lower bound: -0.0144969, upper bound: 0.0144644
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.99
Output dim: 0, lower bound: -0.0146340, upper bound: 0.0148003
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.99
Output dim: 0, lower bound: -0.0146196, upper bound: 0.0148078

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0321665, 0.0323451
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144504, upper bound: 0.0141406
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144485, upper bound: 0.0141406
time: 2.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0322721, 0.0322330
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144409, upper bound: 0.0143914
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144119, upper bound: 0.0144062
time: 2.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0328446, 0.0328039
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144587, upper bound: 0.0146011
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144592, upper bound: 0.0146049
time: 2.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0328621, 0.0327782
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144593, upper bound: 0.0139958
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139394, upper bound: 0.0146746
time: 1.92 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.90 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.90
Output dim: 0, lower bound: -0.0144504, upper bound: 0.0141406
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.90
Output dim: 0, lower bound: -0.0144485, upper bound: 0.0141406
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.90
Output dim: 0, lower bound: -0.0144409, upper bound: 0.0143914
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.90
Output dim: 0, lower bound: -0.0144119, upper bound: 0.0144062
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.90
Output dim: 0, lower bound: -0.0144587, upper bound: 0.0146011
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.90
Output dim: 0, lower bound: -0.0144592, upper bound: 0.0146049
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.90
Output dim: 0, lower bound: -0.0144593, upper bound: 0.0139958
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.90
Output dim: 0, lower bound: -0.0139394, upper bound: 0.0146746

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0319688, 0.0321585
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0143128, upper bound: 0.0134943
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136614, upper bound: 0.0139709
time: 2.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0319799, 0.0321491
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140806, upper bound: 0.0138282
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0141432, upper bound: 0.0137744
time: 2.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0320219, 0.0320760
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0143722, upper bound: 0.0143218
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0143745, upper bound: 0.0143218
time: 2.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0321009, 0.0319827
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0143887, upper bound: 0.0143705
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0143766, upper bound: 0.0143837
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0326278, 0.0326172
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140194, upper bound: 0.0140739
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139400, upper bound: 0.0141832
time: 2.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0326579, 0.0326284
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144307, upper bound: 0.0145838
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144388, upper bound: 0.0145761
time: 2.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0317558, 0.0318434
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144565, upper bound: 0.0139497
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0143483, upper bound: 0.0139930
time: 2.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0319416, 0.0316719
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 135

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129408, upper bound: 0.0136125
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129408, upper bound: 0.0136125
time: 1.88 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.54 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0143128, upper bound: 0.0134943
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0136614, upper bound: 0.0139709
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0140806, upper bound: 0.0138282
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0141432, upper bound: 0.0137744
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0143722, upper bound: 0.0143218
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0143745, upper bound: 0.0143218
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0143887, upper bound: 0.0143705
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0143766, upper bound: 0.0143837
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0140194, upper bound: 0.0140739
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0139400, upper bound: 0.0141832
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0144307, upper bound: 0.0145838
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0144388, upper bound: 0.0145761
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0144565, upper bound: 0.0139497
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0143483, upper bound: 0.0139930
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0129408, upper bound: 0.0136125
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 0, lower bound: -0.0129408, upper bound: 0.0136125

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0308461, 0.0312317
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0119638, upper bound: 0.0115421
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0119638, upper bound: 0.0115421
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0309844, 0.0310358
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135883, upper bound: 0.0138997
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135883, upper bound: 0.0138843
time: 2.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0315559, 0.0315625
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135394, upper bound: 0.0132758
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0133879, upper bound: 0.0132758
time: 2.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0313933, 0.0317335
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 87

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139701, upper bound: 0.0135796
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139076, upper bound: 0.0136053
time: 2.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0319377, 0.0320064
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140192, upper bound: 0.0140162
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140477, upper bound: 0.0139602
time: 2.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0319522, 0.0320003
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0142336, upper bound: 0.0136480
time: 2.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135602, upper bound: 0.0141487
time: 1.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0320068, 0.0319060
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0143798, upper bound: 0.0143300
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0143793, upper bound: 0.0143622
time: 2.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0320246, 0.0318886
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121377, upper bound: 0.0122228
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121377, upper bound: 0.0122228
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0321221, 0.0323582
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136002, upper bound: 0.0136231
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135458, upper bound: 0.0136750
time: 2.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0324167, 0.0321115
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 87

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137608, upper bound: 0.0139029
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137441, upper bound: 0.0139826
time: 2.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0326521, 0.0326221
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144280, upper bound: 0.0144750
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0143110, upper bound: 0.0145811
time: 2.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0326515, 0.0326245
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144361, upper bound: 0.0144564
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0143385, upper bound: 0.0145734
time: 2.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0316887, 0.0317994
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0141012, upper bound: 0.0136247
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0141012, upper bound: 0.0136247
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0317073, 0.0317762
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 87

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0141722, upper bound: 0.0138052
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0141478, upper bound: 0.0138206
time: 2.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0319101, 0.0317030
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126641, upper bound: 0.0132720
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126641, upper bound: 0.0132720
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0319416, 0.0316404
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126641, upper bound: 0.0132720
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126641, upper bound: 0.0132720
time: 1.76 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 6.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0119638, upper bound: 0.0115421
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0119638, upper bound: 0.0115421
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0135883, upper bound: 0.0138997
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0135883, upper bound: 0.0138843
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0135394, upper bound: 0.0132758
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0133879, upper bound: 0.0132758
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0139701, upper bound: 0.0135796
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0139076, upper bound: 0.0136053
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0140192, upper bound: 0.0140162
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0140477, upper bound: 0.0139602
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0142336, upper bound: 0.0136480
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0135602, upper bound: 0.0141487
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0143798, upper bound: 0.0143300
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0143793, upper bound: 0.0143622
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0121377, upper bound: 0.0122228
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0121377, upper bound: 0.0122228
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0136002, upper bound: 0.0136231
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0135458, upper bound: 0.0136750
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0137608, upper bound: 0.0139029
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0137441, upper bound: 0.0139826
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0144280, upper bound: 0.0144750
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0143110, upper bound: 0.0145811
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0144361, upper bound: 0.0144564
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0143385, upper bound: 0.0145734
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0141012, upper bound: 0.0136247
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0141012, upper bound: 0.0136247
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0141722, upper bound: 0.0138052
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0141478, upper bound: 0.0138206
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0126641, upper bound: 0.0132720
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0126641, upper bound: 0.0132720
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0126641, upper bound: 0.0132720
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.13
Output dim: 0, lower bound: -0.0126641, upper bound: 0.0132720

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0309200, 0.0309674
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0108120, upper bound: 0.0108612
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0108120, upper bound: 0.0108612
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0309160, 0.0309493
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0132163, upper bound: 0.0134837
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130593, upper bound: 0.0134837
time: 2.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0315040, 0.0314886
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 87

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0133246, upper bound: 0.0130212
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0132612, upper bound: 0.0130448
time: 2.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0311176, 0.0314737
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139614, upper bound: 0.0135702
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139125, upper bound: 0.0135687
time: 2.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0311335, 0.0314720
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137619, upper bound: 0.0135059
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138092, upper bound: 0.0134823
time: 2.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0315024, 0.0314177
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 87

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140164, upper bound: 0.0139172
time: 2.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139084, upper bound: 0.0140134
time: 2.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0313490, 0.0315527
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139309, upper bound: 0.0138472
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139314, upper bound: 0.0138453
time: 2.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0308383, 0.0310646
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0142247, upper bound: 0.0136026
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135487, upper bound: 0.0136397
time: 2.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0310085, 0.0308864
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135519, upper bound: 0.0134968
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0132159, upper bound: 0.0141410
time: 2.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0318907, 0.0318280
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140148, upper bound: 0.0139094
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139241, upper bound: 0.0139831
time: 2.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0319558, 0.0317900
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0143466, upper bound: 0.0143411
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0143584, upper bound: 0.0143378
time: 2.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0319783, 0.0323129
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135915, upper bound: 0.0130335
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131019, upper bound: 0.0136155
time: 2.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0321221, 0.0322144
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135432, upper bound: 0.0135741
time: 2.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0134842, upper bound: 0.0136724
time: 2.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0321102, 0.0318550
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0133796, upper bound: 0.0134789
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0133346, upper bound: 0.0135001
time: 2.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0321602, 0.0318515
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0133697, upper bound: 0.0135212
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0133312, upper bound: 0.0135558
time: 2.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0325772, 0.0325641
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139862, upper bound: 0.0139399
time: 2.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139109, upper bound: 0.0140703
time: 2.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0325981, 0.0325472
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139719, upper bound: 0.0140289
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138645, upper bound: 0.0142989
time: 1.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0325767, 0.0325662
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 87

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0142551, upper bound: 0.0142194
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0142318, upper bound: 0.0142700
time: 2.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0325977, 0.0325496
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139749, upper bound: 0.0142528
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140340, upper bound: 0.0142092
time: 2.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0316515, 0.0317479
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140925, upper bound: 0.0132266
time: 2.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135058, upper bound: 0.0136161
time: 2.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0316371, 0.0317994
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140003, upper bound: 0.0135237
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140003, upper bound: 0.0135237
time: 2.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0314073, 0.0315106
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0109152, upper bound: 0.0108688
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0109152, upper bound: 0.0108688
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0314417, 0.0314575
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0109152, upper bound: 0.0108688
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0109152, upper bound: 0.0108688
time: 1.00 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.78 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0108120, upper bound: 0.0108612
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0108120, upper bound: 0.0108612
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0132163, upper bound: 0.0134837
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0130593, upper bound: 0.0134837
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0133246, upper bound: 0.0130212
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0132612, upper bound: 0.0130448
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0139614, upper bound: 0.0135702
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0139125, upper bound: 0.0135687
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0137619, upper bound: 0.0135059
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0138092, upper bound: 0.0134823
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0140164, upper bound: 0.0139172
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0139084, upper bound: 0.0140134
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0139309, upper bound: 0.0138472
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0139314, upper bound: 0.0138453
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0142247, upper bound: 0.0136026
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0135487, upper bound: 0.0136397
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0135519, upper bound: 0.0134968
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0132159, upper bound: 0.0141410
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0140148, upper bound: 0.0139094
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0139241, upper bound: 0.0139831
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0143466, upper bound: 0.0143411
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0143584, upper bound: 0.0143378
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0135915, upper bound: 0.0130335
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0131019, upper bound: 0.0136155
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0135432, upper bound: 0.0135741
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0134842, upper bound: 0.0136724
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0133796, upper bound: 0.0134789
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0133346, upper bound: 0.0135001
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0133697, upper bound: 0.0135212
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0133312, upper bound: 0.0135558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0139862, upper bound: 0.0139399
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0139109, upper bound: 0.0140703
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0139719, upper bound: 0.0140289
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0138645, upper bound: 0.0142989
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0142551, upper bound: 0.0142194
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0142318, upper bound: 0.0142700
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0139749, upper bound: 0.0142528
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0140340, upper bound: 0.0142092
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0140925, upper bound: 0.0132266
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0135058, upper bound: 0.0136161
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0140003, upper bound: 0.0135237
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0140003, upper bound: 0.0135237
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0109152, upper bound: 0.0108688
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0109152, upper bound: 0.0108688
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0109152, upper bound: 0.0108688
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 0, lower bound: -0.0109152, upper bound: 0.0108688

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0308769, 0.0308990
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123218, upper bound: 0.0125018
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123218, upper bound: 0.0125018
time: 2.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0308657, 0.0309493
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 87

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129692, upper bound: 0.0132005
time: 2.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129434, upper bound: 0.0132252
time: 2.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0310121, 0.0314406
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136565, upper bound: 0.0132531
time: 2.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136564, upper bound: 0.0132531
time: 2.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0310354, 0.0313682
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136114, upper bound: 0.0132520
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136114, upper bound: 0.0132521
time: 2.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0309979, 0.0313340
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137591, upper bound: 0.0133921
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136706, upper bound: 0.0135031
time: 2.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0309955, 0.0313585
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0109870, upper bound: 0.0108885
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0109870, upper bound: 0.0108885
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0314290, 0.0313627
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136865, upper bound: 0.0135129
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0134358, upper bound: 0.0135794
time: 2.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0314454, 0.0313443
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0133677, upper bound: 0.0134638
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0133677, upper bound: 0.0134634
time: 2.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0312201, 0.0314046
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136377, upper bound: 0.0133321
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0133697, upper bound: 0.0135586
time: 2.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0312009, 0.0314295
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137957, upper bound: 0.0137384
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138239, upper bound: 0.0136992
time: 2.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0307190, 0.0309834
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138867, upper bound: 0.0132159
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0131381, upper bound: 0.0132648
time: 2.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0307817, 0.0309454
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0141085, upper bound: 0.0134514
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139919, upper bound: 0.0135093
time: 2.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0293217, 0.0295270
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0134311, upper bound: 0.0133823
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0134336, upper bound: 0.0133797
time: 2.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0297690, 0.0291996
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 87

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130452, upper bound: 0.0139294
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130441, upper bound: 0.0139598
time: 2.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0317514, 0.0317921
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138443, upper bound: 0.0136604
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137645, upper bound: 0.0137562
time: 2.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0318907, 0.0316886
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 135

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0134599, upper bound: 0.0134069
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0133650, upper bound: 0.0134760
time: 2.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0319487, 0.0317817
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 87

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0141581, upper bound: 0.0141348
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140808, upper bound: 0.0141658
time: 2.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0319475, 0.0317829
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128551, upper bound: 0.0128525
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128551, upper bound: 0.0128525
time: 1.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0302334, 0.0309068
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0134148, upper bound: 0.0124997
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129284, upper bound: 0.0128248
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0305835, 0.0305680
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130055, upper bound: 0.0135345
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130055, upper bound: 0.0135345
time: 2.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0320473, 0.0321534
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0133963, upper bound: 0.0134694
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0134420, upper bound: 0.0134352
time: 2.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0320665, 0.0321394
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0134256, upper bound: 0.0135847
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0134166, upper bound: 0.0136178
time: 2.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0319664, 0.0318214
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120069, upper bound: 0.0120523
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120069, upper bound: 0.0120523
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0321102, 0.0317113
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0133262, upper bound: 0.0134774
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0133123, upper bound: 0.0134920
time: 2.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0320164, 0.0317882
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0131909, upper bound: 0.0133511
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0131909, upper bound: 0.0133511
time: 2.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0321602, 0.0317077
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129771, upper bound: 0.0132498
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0130320, upper bound: 0.0132106
time: 2.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0320703, 0.0322584
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139775, upper bound: 0.0133266
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0134486, upper bound: 0.0139322
time: 2.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0323883, 0.0320572
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139024, upper bound: 0.0140356
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138894, upper bound: 0.0140619
time: 2.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0317464, 0.0318151
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0118906, upper bound: 0.0119603
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0118906, upper bound: 0.0119603
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0319117, 0.0316955
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0110366, upper bound: 0.0110710
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0110366, upper bound: 0.0110710
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0322758, 0.0323112
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 135

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129990, upper bound: 0.0130151
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129990, upper bound: 0.0130151
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0323216, 0.0323400
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136262, upper bound: 0.0136676
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136281, upper bound: 0.0136676
time: 2.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0321496, 0.0319618
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136421, upper bound: 0.0138375
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135674, upper bound: 0.0139207
time: 2.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0320099, 0.0321876
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137022, upper bound: 0.0139030
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137022, upper bound: 0.0139031
time: 2.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0298884, 0.0303777
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 87

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138783, upper bound: 0.0130304
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138469, upper bound: 0.0130304
time: 2.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0302293, 0.0299848
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0134002, upper bound: 0.0134960
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0134014, upper bound: 0.0134923
time: 2.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0315657, 0.0317340
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139741, upper bound: 0.0135016
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139776, upper bound: 0.0134583
time: 2.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0315720, 0.0317323
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139741, upper bound: 0.0135016
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139776, upper bound: 0.0134583
time: 2.13 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.16 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0123218, upper bound: 0.0125018
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0123218, upper bound: 0.0125018
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0129692, upper bound: 0.0132005
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0129434, upper bound: 0.0132252
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0136565, upper bound: 0.0132531
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0136564, upper bound: 0.0132531
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0136114, upper bound: 0.0132520
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0136114, upper bound: 0.0132521
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0137591, upper bound: 0.0133921
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0136706, upper bound: 0.0135031
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0109870, upper bound: 0.0108885
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0109870, upper bound: 0.0108885
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0136865, upper bound: 0.0135129
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0134358, upper bound: 0.0135794
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0133677, upper bound: 0.0134638
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0133677, upper bound: 0.0134634
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0136377, upper bound: 0.0133321
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0133697, upper bound: 0.0135586
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0137957, upper bound: 0.0137384
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0138239, upper bound: 0.0136992
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0138867, upper bound: 0.0132159
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0131381, upper bound: 0.0132648
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0141085, upper bound: 0.0134514
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0139919, upper bound: 0.0135093
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0134311, upper bound: 0.0133823
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0134336, upper bound: 0.0133797
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0130452, upper bound: 0.0139294
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0130441, upper bound: 0.0139598
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0138443, upper bound: 0.0136604
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0137645, upper bound: 0.0137562
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0134599, upper bound: 0.0134069
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0133650, upper bound: 0.0134760
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0141581, upper bound: 0.0141348
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0140808, upper bound: 0.0141658
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0128551, upper bound: 0.0128525
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0128551, upper bound: 0.0128525
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0134148, upper bound: 0.0124997
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0129284, upper bound: 0.0128248
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0130055, upper bound: 0.0135345
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0130055, upper bound: 0.0135345
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0133963, upper bound: 0.0134694
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0134420, upper bound: 0.0134352
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0134256, upper bound: 0.0135847
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0134166, upper bound: 0.0136178
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0120069, upper bound: 0.0120523
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0120069, upper bound: 0.0120523
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0133262, upper bound: 0.0134774
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0133123, upper bound: 0.0134920
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0131909, upper bound: 0.0133511
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0131909, upper bound: 0.0133511
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0129771, upper bound: 0.0132498
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0130320, upper bound: 0.0132106
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0139775, upper bound: 0.0133266
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0134486, upper bound: 0.0139322
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0139024, upper bound: 0.0140356
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0138894, upper bound: 0.0140619
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0118906, upper bound: 0.0119603
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0118906, upper bound: 0.0119603
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0110366, upper bound: 0.0110710
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0110366, upper bound: 0.0110710
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0129990, upper bound: 0.0130151
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0129990, upper bound: 0.0130151
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0136262, upper bound: 0.0136676
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0136281, upper bound: 0.0136676
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0136421, upper bound: 0.0138375
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0135674, upper bound: 0.0139207
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0137022, upper bound: 0.0139030
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0137022, upper bound: 0.0139031
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0138783, upper bound: 0.0130304
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0138469, upper bound: 0.0130304
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0134002, upper bound: 0.0134960
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0134014, upper bound: 0.0134923
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0139741, upper bound: 0.0135016
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0139776, upper bound: 0.0134583
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0139741, upper bound: 0.0135016
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0139776, upper bound: 0.0134583

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0309261, 0.0314000
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135088, upper bound: 0.0126457
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128931, upper bound: 0.0130697
time: 2.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0310121, 0.0313546
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129074, upper bound: 0.0125045
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129074, upper bound: 0.0125045
time: 1.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0309494, 0.0313220
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136086, upper bound: 0.0131464
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135070, upper bound: 0.0132493
time: 2.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0310354, 0.0312822
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123463, upper bound: 0.0120956
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123463, upper bound: 0.0120956
time: 2.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0309246, 0.0312729
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129757, upper bound: 0.0126236
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129757, upper bound: 0.0126236
time: 2.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0309458, 0.0312608
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0105064, upper bound: 0.0104537
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0105064, upper bound: 0.0104537
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0312233, 0.0312527
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0131730, upper bound: 0.0129958
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0130813, upper bound: 0.0130370
time: 2.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0314290, 0.0311569
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 87

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0133720, upper bound: 0.0133740
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0133021, upper bound: 0.0133870
time: 2.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0313999, 0.0312691
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0133569, upper bound: 0.0129223
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128305, upper bound: 0.0134531
time: 2.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0313701, 0.0313443
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 135

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0133594, upper bound: 0.0134322
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0133594, upper bound: 0.0134551
time: 2.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0303334, 0.0306329
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0134949, upper bound: 0.0127221
time: 2.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128134, upper bound: 0.0131433
time: 2.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0304580, 0.0305179
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125058, upper bound: 0.0125999
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125058, upper bound: 0.0125999
time: 2.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0310708, 0.0312947
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0115264, upper bound: 0.0115500
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0115264, upper bound: 0.0115500
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0310661, 0.0313117
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138159, upper bound: 0.0130715
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0132784, upper bound: 0.0136911
time: 2.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0305137, 0.0308832
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138640, upper bound: 0.0131592
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138605, upper bound: 0.0131937
time: 2.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0306860, 0.0308655
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121771, upper bound: 0.0119029
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121771, upper bound: 0.0119030
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0306987, 0.0308497
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 135

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128601, upper bound: 0.0124237
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128601, upper bound: 0.0124237
time: 1.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0291874, 0.0293826
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0109222, upper bound: 0.0109630
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0109222, upper bound: 0.0109630
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0291774, 0.0294012
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0133835, upper bound: 0.0133593
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0134120, upper bound: 0.0133341
time: 2.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0294099, 0.0289302
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126859, upper bound: 0.0136213
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127664, upper bound: 0.0135821
time: 2.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0294996, 0.0288818
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126839, upper bound: 0.0136519
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127657, upper bound: 0.0136148
time: 2.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0316572, 0.0317071
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 135

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124391, upper bound: 0.0124184
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124391, upper bound: 0.0124184
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0316692, 0.0316979
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0131279, upper bound: 0.0131532
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0131232, upper bound: 0.0131602
time: 2.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747
1: -0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919
2: -0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677
3: -0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820
4: -0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227
5: -0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0313761, 0.0314303
6: -0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399
7: -0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872
8: -0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206
9: -0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120902, upper bound: 0.0120404
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120902, upper bound: 0.0120404
time: 1.59 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 5.82 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0135088, upper bound: 0.0126457
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0128931, upper bound: 0.0130697
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0129074, upper bound: 0.0125045
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0129074, upper bound: 0.0125045
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0136086, upper bound: 0.0131464
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0135070, upper bound: 0.0132493
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0123463, upper bound: 0.0120956
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0123463, upper bound: 0.0120956
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0129757, upper bound: 0.0126236
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0129757, upper bound: 0.0126236
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0105064, upper bound: 0.0104537
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0105064, upper bound: 0.0104537
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0131730, upper bound: 0.0129958
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0130813, upper bound: 0.0130370
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0133720, upper bound: 0.0133740
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0133021, upper bound: 0.0133870
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0133569, upper bound: 0.0129223
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0128305, upper bound: 0.0134531
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0133594, upper bound: 0.0134322
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0133594, upper bound: 0.0134551
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0134949, upper bound: 0.0127221
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0128134, upper bound: 0.0131433
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0125058, upper bound: 0.0125999
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0125058, upper bound: 0.0125999
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0115264, upper bound: 0.0115500
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0115264, upper bound: 0.0115500
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0138159, upper bound: 0.0130715
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0132784, upper bound: 0.0136911
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0138640, upper bound: 0.0131592
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0138605, upper bound: 0.0131937
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0121771, upper bound: 0.0119029
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0121771, upper bound: 0.0119030
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0128601, upper bound: 0.0124237
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0128601, upper bound: 0.0124237
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0109222, upper bound: 0.0109630
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0109222, upper bound: 0.0109630
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0133835, upper bound: 0.0133593
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0134120, upper bound: 0.0133341
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0126859, upper bound: 0.0136213
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0127664, upper bound: 0.0135821
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0126839, upper bound: 0.0136519
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0127657, upper bound: 0.0136148
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0124391, upper bound: 0.0124184
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0124391, upper bound: 0.0124184
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0131279, upper bound: 0.0131532
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0131232, upper bound: 0.0131602
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0120902, upper bound: 0.0120404
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 0, lower bound: -0.0120902, upper bound: 0.0120404
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0133650, upper bound: 0.0134760
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0141581, upper bound: 0.0141348
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0140808, upper bound: 0.0141658
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0130055, upper bound: 0.0135345
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0130055, upper bound: 0.0135345
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0133963, upper bound: 0.0134694
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0134420, upper bound: 0.0134352
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0134256, upper bound: 0.0135847
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0134166, upper bound: 0.0136178
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0133262, upper bound: 0.0134774
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0133123, upper bound: 0.0134920
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0139775, upper bound: 0.0133266
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0134486, upper bound: 0.0139322
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0139024, upper bound: 0.0140356
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0138894, upper bound: 0.0140619
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0136262, upper bound: 0.0136676
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0136281, upper bound: 0.0136676
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0136421, upper bound: 0.0138375
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0135674, upper bound: 0.0139207
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0137022, upper bound: 0.0139030
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0137022, upper bound: 0.0139031
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0138783, upper bound: 0.0130304
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0138469, upper bound: 0.0130304
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0134002, upper bound: 0.0134960
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0134014, upper bound: 0.0134923
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0139741, upper bound: 0.0135016
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0139776, upper bound: 0.0134583
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0139741, upper bound: 0.0135016
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 0, lower bound: -0.0139776, upper bound: 0.0134583

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.05 + 597.21 = 601.26 seconds
