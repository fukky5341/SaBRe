## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.020861730000000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693)
1: (-0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900)
2: (0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209)
3: (-0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987)
4: (0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823)
5: (-0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120)
6: (-0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657)
7: (-0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077)
8: (-0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964)
9: (-0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.87 + 3.37 = 4.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0231797, upper bound: 0.0231797

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0230220, upper bound: 0.0226004
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0226004, upper bound: 0.0230220
time: 1.58 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.08 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.08
Output dim: 4, lower bound: -0.0230220, upper bound: 0.0226004
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.08
Output dim: 4, lower bound: -0.0226004, upper bound: 0.0230220

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0229729, upper bound: 0.0225143
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0229641, upper bound: 0.0225509
time: 1.58 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0225830, upper bound: 0.0228508
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0224987, upper bound: 0.0230056
time: 2.24 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.59 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 4, lower bound: -0.0229729, upper bound: 0.0225143
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 4, lower bound: -0.0229641, upper bound: 0.0225509
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 4, lower bound: -0.0225830, upper bound: 0.0228508
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 4, lower bound: -0.0224987, upper bound: 0.0230056

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0227880, upper bound: 0.0219746
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0224158, upper bound: 0.0223238
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0229641, upper bound: 0.0222675
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0225695, upper bound: 0.0225509
time: 2.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0225830, upper bound: 0.0223892
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0222687, upper bound: 0.0228509
time: 2.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0224534, upper bound: 0.0229646
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0224584, upper bound: 0.0229531
time: 1.58 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.33 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 4, lower bound: -0.0227880, upper bound: 0.0219746
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 4, lower bound: -0.0224158, upper bound: 0.0223238
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 4, lower bound: -0.0229641, upper bound: 0.0222675
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 4, lower bound: -0.0225695, upper bound: 0.0225509
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 4, lower bound: -0.0225830, upper bound: 0.0223892
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 4, lower bound: -0.0222687, upper bound: 0.0228509
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 4, lower bound: -0.0224534, upper bound: 0.0229646
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.33
Output dim: 4, lower bound: -0.0224584, upper bound: 0.0229531

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0227360, upper bound: 0.0219348
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0227473, upper bound: 0.0219342
time: 2.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 129

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0217760, upper bound: 0.0210844
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0210909, upper bound: 0.0216306
time: 2.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0204904, upper bound: 0.0201525
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0204904, upper bound: 0.0201525
time: 1.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0225153, upper bound: 0.0225113
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0225284, upper bound: 0.0225080
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0224922, upper bound: 0.0223408
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0225327, upper bound: 0.0222766
time: 4.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 129

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0215749, upper bound: 0.0215814
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0210683, upper bound: 0.0221959
time: 1.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 218

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0221507, upper bound: 0.0225250
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0219837, upper bound: 0.0226530
time: 2.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0215847, upper bound: 0.0215094
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0211219, upper bound: 0.0221040
time: 2.31 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.73 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0227360, upper bound: 0.0219348
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0227473, upper bound: 0.0219342
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0217760, upper bound: 0.0210844
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0210909, upper bound: 0.0216306
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0204904, upper bound: 0.0201525
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0204904, upper bound: 0.0201525
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0225153, upper bound: 0.0225113
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0225284, upper bound: 0.0225080
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0224922, upper bound: 0.0223408
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0225327, upper bound: 0.0222766
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0215749, upper bound: 0.0215814
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0210683, upper bound: 0.0221959
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0221507, upper bound: 0.0225250
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0219837, upper bound: 0.0226530
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0215847, upper bound: 0.0215094
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 4, lower bound: -0.0211219, upper bound: 0.0221040

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0196899, upper bound: 0.0190909
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0196899, upper bound: 0.0190909
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0226337, upper bound: 0.0218869
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0226980, upper bound: 0.0217379
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082865
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0295218, 0.0303609

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0217712, upper bound: 0.0210643
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0216565, upper bound: 0.0210675
time: 2.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0298039

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0200188, upper bound: 0.0201796
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0195673, upper bound: 0.0204932
time: 1.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0224977, upper bound: 0.0224086
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0224596, upper bound: 0.0224938
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0216862, upper bound: 0.0211297
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0212020, upper bound: 0.0216291
time: 1.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0202356, upper bound: 0.0202544
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0202356, upper bound: 0.0202544
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0221172, upper bound: 0.0218922
time: 2.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0221441, upper bound: 0.0218320
time: 1.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0215264, upper bound: 0.0215282
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0215162, upper bound: 0.0215304
time: 1.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0298850

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0201588, upper bound: 0.0207973
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0196811, upper bound: 0.0212119
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220367, upper bound: 0.0216071
time: 2.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0213492, upper bound: 0.0224155
time: 2.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0218704, upper bound: 0.0217592
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0211827, upper bound: 0.0225421
time: 2.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0215795, upper bound: 0.0214099
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0215403, upper bound: 0.0215044
time: 2.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0204757, upper bound: 0.0220398
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0204757, upper bound: 0.0214499
time: 2.02 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.91 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0196899, upper bound: 0.0190909
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0196899, upper bound: 0.0190909
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0226337, upper bound: 0.0218869
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0226980, upper bound: 0.0217379
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0217712, upper bound: 0.0210643
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0216565, upper bound: 0.0210675
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0200188, upper bound: 0.0201796
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0195673, upper bound: 0.0204932
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0224977, upper bound: 0.0224086
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0224596, upper bound: 0.0224938
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0216862, upper bound: 0.0211297
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0212020, upper bound: 0.0216291
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0202356, upper bound: 0.0202544
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0202356, upper bound: 0.0202544
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0221172, upper bound: 0.0218922
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0221441, upper bound: 0.0218320
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0215264, upper bound: 0.0215282
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0215162, upper bound: 0.0215304
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0201588, upper bound: 0.0207973
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0196811, upper bound: 0.0212119
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0220367, upper bound: 0.0216071
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0213492, upper bound: 0.0224155
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0218704, upper bound: 0.0217592
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0211827, upper bound: 0.0225421
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0215795, upper bound: 0.0214099
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0215403, upper bound: 0.0215044
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0204757, upper bound: 0.0220398
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 4, lower bound: -0.0204757, upper bound: 0.0214499

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0225597, upper bound: 0.0217049
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0222177, upper bound: 0.0218114
time: 1.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0222161, upper bound: 0.0213594
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0223069, upper bound: 0.0213182
time: 2.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082796
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0294913, 0.0303609

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0216682, upper bound: 0.0204603
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0209292, upper bound: 0.0209493
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082861
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0295185, 0.0303609

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0204879, upper bound: 0.0196545
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0200773, upper bound: 0.0199507
time: 2.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0222470, upper bound: 0.0222642
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0223564, upper bound: 0.0222065
time: 2.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220220, upper bound: 0.0221058
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220723, upper bound: 0.0220784
time: 2.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0216810, upper bound: 0.0210629
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0216109, upper bound: 0.0211246
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0211380, upper bound: 0.0212665
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0207510, upper bound: 0.0215646
time: 2.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0221121, upper bound: 0.0218600
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220690, upper bound: 0.0218870
time: 2.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0221226, upper bound: 0.0217890
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220763, upper bound: 0.0218137
time: 2.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0164400, upper bound: 0.0164094
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0164400, upper bound: 0.0164094
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0214102, upper bound: 0.0208313
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0206822, upper bound: 0.0214177
time: 2.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082625, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0294208

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0196811, upper bound: 0.0209737
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0196325, upper bound: 0.0212119
time: 2.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0219466, upper bound: 0.0212408
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0216904, upper bound: 0.0215215
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0300746

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0210988, upper bound: 0.0223674
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0213021, upper bound: 0.0222973
time: 2.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0211817, upper bound: 0.0217039
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0218138, upper bound: 0.0210064
time: 1.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0299555

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0164888, upper bound: 0.0170786
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0164888, upper bound: 0.0170786
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0214651, upper bound: 0.0204834
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0208095, upper bound: 0.0212974
time: 1.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0162552, upper bound: 0.0163141
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0162552, upper bound: 0.0163141
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0081629, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0301367, 0.0289131

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0204259, upper bound: 0.0219880
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0204190, upper bound: 0.0219945
time: 2.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082589, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0297600, 0.0293202

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0197926, upper bound: 0.0196892
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0196013, upper bound: 0.0198717
time: 1.98 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.97 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0225597, upper bound: 0.0217049
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0222177, upper bound: 0.0218114
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0222161, upper bound: 0.0213594
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0223069, upper bound: 0.0213182
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0216682, upper bound: 0.0204603
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0209292, upper bound: 0.0209493
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0204879, upper bound: 0.0196545
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0200773, upper bound: 0.0199507
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0222470, upper bound: 0.0222642
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0223564, upper bound: 0.0222065
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0220220, upper bound: 0.0221058
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0220723, upper bound: 0.0220784
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0216810, upper bound: 0.0210629
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0216109, upper bound: 0.0211246
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0211380, upper bound: 0.0212665
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0207510, upper bound: 0.0215646
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0221121, upper bound: 0.0218600
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0220690, upper bound: 0.0218870
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0221226, upper bound: 0.0217890
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0220763, upper bound: 0.0218137
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0164400, upper bound: 0.0164094
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0164400, upper bound: 0.0164094
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0214102, upper bound: 0.0208313
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0206822, upper bound: 0.0214177
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0196811, upper bound: 0.0209737
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0196325, upper bound: 0.0212119
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0219466, upper bound: 0.0212408
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0216904, upper bound: 0.0215215
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0210988, upper bound: 0.0223674
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0213021, upper bound: 0.0222973
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0211817, upper bound: 0.0217039
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0218138, upper bound: 0.0210064
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0164888, upper bound: 0.0170786
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0164888, upper bound: 0.0170786
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0214651, upper bound: 0.0204834
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0208095, upper bound: 0.0212974
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0162552, upper bound: 0.0163141
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0162552, upper bound: 0.0163141
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0204259, upper bound: 0.0219880
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0204190, upper bound: 0.0219945
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0197926, upper bound: 0.0196892
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 4, lower bound: -0.0196013, upper bound: 0.0198717

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0302586, 0.0303609

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0225597, upper bound: 0.0216024
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0221086, upper bound: 0.0217049
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0222016, upper bound: 0.0216869
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0219512, upper bound: 0.0217938
time: 2.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0299926, 0.0303609

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0191726, upper bound: 0.0184642
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0191726, upper bound: 0.0184642
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0299146, 0.0303609

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0212703, upper bound: 0.0198908
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0207868, upper bound: 0.0201605
time: 2.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0081468
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0289221, 0.0303609

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0216682, upper bound: 0.0202649
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0212618, upper bound: 0.0204603
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082557
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0293839, 0.0301576

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0209119, upper bound: 0.0208839
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0207398, upper bound: 0.0209345
time: 2.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0221752, upper bound: 0.0219392
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0219030, upper bound: 0.0221767
time: 2.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0223321, upper bound: 0.0221291
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0222788, upper bound: 0.0221779
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0219506, upper bound: 0.0218281
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0216348, upper bound: 0.0220239
time: 1.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220570, upper bound: 0.0220082
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220080, upper bound: 0.0220557
time: 2.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303436, 0.0303609

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0215030, upper bound: 0.0204908
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0211305, upper bound: 0.0208767
time: 2.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0180723, upper bound: 0.0179528
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0180723, upper bound: 0.0179528
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0302710, 0.0303190

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0211173, upper bound: 0.0212365
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0210600, upper bound: 0.0212589
time: 1.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0300769

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0205405, upper bound: 0.0207987
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0199874, upper bound: 0.0214513
time: 2.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0214662, upper bound: 0.0218034
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220546, upper bound: 0.0212425
time: 1.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0213829, upper bound: 0.0218307
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220126, upper bound: 0.0212642
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 129

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0214355, upper bound: 0.0205804
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0208591, upper bound: 0.0211303
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0219963, upper bound: 0.0216368
time: 2.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0218350, upper bound: 0.0217522
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0300471, 0.0303609

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0202668, upper bound: 0.0193413
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0200284, upper bound: 0.0197612
time: 2.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0300711

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0202933, upper bound: 0.0212817
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0205468, upper bound: 0.0211375
time: 2.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0295937

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0196151, upper bound: 0.0205586
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0193757, upper bound: 0.0209114
time: 2.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082432, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0293641

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0150033, upper bound: 0.0157111
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0150033, upper bound: 0.0157111
time: 2.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0192178, upper bound: 0.0184618
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0192178, upper bound: 0.0184618
time: 2.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0189912, upper bound: 0.0187185
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0189912, upper bound: 0.0187185
time: 1.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0299384

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0202520, upper bound: 0.0209789
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0196822, upper bound: 0.0214826
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0299648

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0201746, upper bound: 0.0207961
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0199375, upper bound: 0.0212420
time: 2.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0081382, 0.0082643
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0293735, 0.0288386

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0191662, upper bound: 0.0193024
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0191662, upper bound: 0.0193024
time: 2.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082212, 0.0081803
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0290171, 0.0291904

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0217633, upper bound: 0.0209502
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0217415, upper bound: 0.0209557
time: 2.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0300552, 0.0303609

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0212858, upper bound: 0.0200956
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0208893, upper bound: 0.0202961
time: 2.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0300933

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0193092, upper bound: 0.0197404
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0191921, upper bound: 0.0200801
time: 2.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0081315, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0299955, 0.0287773

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 145

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0190686, upper bound: 0.0201230
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0190686, upper bound: 0.0201230
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0081283, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0299892, 0.0287638

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0191253, upper bound: 0.0202586
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0189086, upper bound: 0.0204846
time: 2.21 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.47 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0225597, upper bound: 0.0216024
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0221086, upper bound: 0.0217049
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0222016, upper bound: 0.0216869
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0219512, upper bound: 0.0217938
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0191726, upper bound: 0.0184642
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0191726, upper bound: 0.0184642
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0212703, upper bound: 0.0198908
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0207868, upper bound: 0.0201605
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0216682, upper bound: 0.0202649
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0212618, upper bound: 0.0204603
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0209119, upper bound: 0.0208839
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0207398, upper bound: 0.0209345
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0221752, upper bound: 0.0219392
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0219030, upper bound: 0.0221767
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0223321, upper bound: 0.0221291
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0222788, upper bound: 0.0221779
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0219506, upper bound: 0.0218281
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0216348, upper bound: 0.0220239
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0220570, upper bound: 0.0220082
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0220080, upper bound: 0.0220557
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0215030, upper bound: 0.0204908
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0211305, upper bound: 0.0208767
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0180723, upper bound: 0.0179528
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0180723, upper bound: 0.0179528
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0211173, upper bound: 0.0212365
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0210600, upper bound: 0.0212589
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0205405, upper bound: 0.0207987
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0199874, upper bound: 0.0214513
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0214662, upper bound: 0.0218034
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0220546, upper bound: 0.0212425
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0213829, upper bound: 0.0218307
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0220126, upper bound: 0.0212642
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0214355, upper bound: 0.0205804
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0208591, upper bound: 0.0211303
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0219963, upper bound: 0.0216368
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0218350, upper bound: 0.0217522
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0202668, upper bound: 0.0193413
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0200284, upper bound: 0.0197612
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0202933, upper bound: 0.0212817
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0205468, upper bound: 0.0211375
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0196151, upper bound: 0.0205586
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0193757, upper bound: 0.0209114
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0150033, upper bound: 0.0157111
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0150033, upper bound: 0.0157111
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0192178, upper bound: 0.0184618
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0192178, upper bound: 0.0184618
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0189912, upper bound: 0.0187185
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0189912, upper bound: 0.0187185
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0202520, upper bound: 0.0209789
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0196822, upper bound: 0.0214826
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0201746, upper bound: 0.0207961
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0199375, upper bound: 0.0212420
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0191662, upper bound: 0.0193024
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0191662, upper bound: 0.0193024
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0217633, upper bound: 0.0209502
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0217415, upper bound: 0.0209557
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0212858, upper bound: 0.0200956
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0208893, upper bound: 0.0202961
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0193092, upper bound: 0.0197404
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0191921, upper bound: 0.0200801
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0190686, upper bound: 0.0201230
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0190686, upper bound: 0.0201230
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0191253, upper bound: 0.0202586
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.47
Output dim: 4, lower bound: -0.0189086, upper bound: 0.0204846

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0201089, upper bound: 0.0195127
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0201089, upper bound: 0.0195127
time: 1.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0221034, upper bound: 0.0216586
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220938, upper bound: 0.0216998
time: 2.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0222016, upper bound: 0.0215191
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220131, upper bound: 0.0216869
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0219462, upper bound: 0.0217498
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0218807, upper bound: 0.0217887
time: 2.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082524
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0293421, 0.0303609

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0212545, upper bound: 0.0198313
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0210979, upper bound: 0.0198737
time: 1.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0081705
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0291043, 0.0303609

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0185083, upper bound: 0.0176038
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0185083, upper bound: 0.0176038
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082761
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0295521, 0.0303609

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0201146, upper bound: 0.0190011
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0198215, upper bound: 0.0193650
time: 2.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082539
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0293764, 0.0301528

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0208742, upper bound: 0.0208573
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0208732, upper bound: 0.0208550
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082543
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0293780, 0.0301500

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0207015, upper bound: 0.0209090
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0207037, upper bound: 0.0209075
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303508, 0.0303609

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220905, upper bound: 0.0218896
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0221238, upper bound: 0.0218420
time: 2.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0214711, upper bound: 0.0217964
time: 2.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0214997, upper bound: 0.0217613
time: 2.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0302965, 0.0303609

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 129

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0216315, upper bound: 0.0208532
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0212099, upper bound: 0.0214470
time: 2.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303303, 0.0303609

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0217895, upper bound: 0.0217022
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0218238, upper bound: 0.0216923
time: 2.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0208917, upper bound: 0.0203959
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0205687, upper bound: 0.0206748
time: 2.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0209792, upper bound: 0.0219655
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0215755, upper bound: 0.0213503
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303605, 0.0303609

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220520, upper bound: 0.0219567
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0219591, upper bound: 0.0220031
time: 2.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0219188, upper bound: 0.0220037
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0219583, upper bound: 0.0219638
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0300449, 0.0303609

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0160263, upper bound: 0.0157435
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0160263, upper bound: 0.0157435
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0301821, 0.0303609

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0211305, upper bound: 0.0206240
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0208791, upper bound: 0.0208767
time: 1.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0301535, 0.0302436

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0204381, upper bound: 0.0211737
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0210560, upper bound: 0.0206542
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0301863, 0.0302054

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0207711, upper bound: 0.0211153
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0209077, upper bound: 0.0209852
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082845, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0294697

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0199874, upper bound: 0.0211137
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0198352, upper bound: 0.0214513
time: 2.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082833, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0298368, 0.0294511

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0214332, upper bound: 0.0217643
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0214130, upper bound: 0.0217595
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082866
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0294651, 0.0298615

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220145, upper bound: 0.0211830
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220159, upper bound: 0.0212083
time: 2.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082773, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0298580, 0.0294254

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0192968, upper bound: 0.0197451
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0192968, upper bound: 0.0197451
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0294905, 0.0298393

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0218937, upper bound: 0.0202939
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0212105, upper bound: 0.0211562
time: 2.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0296021, 0.0303609

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0213716, upper bound: 0.0202567
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0210650, upper bound: 0.0205142
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0297648

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0208176, upper bound: 0.0210904
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0208214, upper bound: 0.0210873
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 129

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0212945, upper bound: 0.0203523
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0207295, upper bound: 0.0209835
time: 2.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0206730, upper bound: 0.0204486
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0204290, upper bound: 0.0206855
time: 2.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082897, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0300663, 0.0296209

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 218

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0200295, upper bound: 0.0207515
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0199445, upper bound: 0.0209972
time: 2.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0300130, 0.0296742

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0200431, upper bound: 0.0206337
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0200483, upper bound: 0.0206337
time: 2.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0081839, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0290620

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 218

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0190659, upper bound: 0.0203501
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0189170, upper bound: 0.0206274
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0302837, 0.0298532

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0202388, upper bound: 0.0208519
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0202100, upper bound: 0.0209467
time: 1.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082565, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0294599

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0152081, upper bound: 0.0159210
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0152081, upper bound: 0.0159210
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082459, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0294000

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0199339, upper bound: 0.0210827
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0198984, upper bound: 0.0212397
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0081939, 0.0081541
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0288792, 0.0290482

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0217316, upper bound: 0.0208784
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0216955, upper bound: 0.0209150
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0081952, 0.0081530
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0288748, 0.0290537

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0217364, upper bound: 0.0209252
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0217118, upper bound: 0.0209507
time: 1.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693
1: -0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900
2: 0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209
3: -0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987
4: 0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823
5: -0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120
6: -0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657
7: -0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077
8: -0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964
9: -0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0298021, 0.0303609

Time for backsubstitution: 0.88 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.24 + 596.56 = 600.80 seconds
