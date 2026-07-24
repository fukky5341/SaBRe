## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.006503490000000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142)
1: (0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415)
2: (-0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272)
3: (-0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832)
4: (0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747)
5: (-0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980)
6: (0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015)
7: (-0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0243378, 0.0243378)
8: (-0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058)
9: (-0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.92 + 3.68 = 5.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0069930, upper bound: 0.0069925

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0068499, upper bound: 0.0068415
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0068420, upper bound: 0.0068499
time: 2.42 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.35 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.35
Output dim: 6, lower bound: -0.0068499, upper bound: 0.0068415
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.35
Output dim: 6, lower bound: -0.0068420, upper bound: 0.0068499

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0243232, 0.0243217
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0068060, upper bound: 0.0067536
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067598, upper bound: 0.0067978
time: 2.71 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0243217, 0.0243232
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067990, upper bound: 0.0067593
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067549, upper bound: 0.0068060
time: 2.89 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 7.78 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 7.78
Output dim: 6, lower bound: -0.0068060, upper bound: 0.0067536
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 7.78
Output dim: 6, lower bound: -0.0067598, upper bound: 0.0067978
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 7.78
Output dim: 6, lower bound: -0.0067990, upper bound: 0.0067593
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 7.78
Output dim: 6, lower bound: -0.0067549, upper bound: 0.0068060

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0239293, 0.0238117
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062372, upper bound: 0.0062126
time: 2.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062372, upper bound: 0.0062126
time: 2.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0238132, 0.0239270
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062126, upper bound: 0.0062367
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062135, upper bound: 0.0062380
time: 2.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0239270, 0.0238132
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062372, upper bound: 0.0062121
time: 2.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062372, upper bound: 0.0062121
time: 2.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0238117, 0.0239293
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062126, upper bound: 0.0062367
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062126, upper bound: 0.0062367
time: 2.22 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 6.78 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 6.78
Output dim: 6, lower bound: -0.0062372, upper bound: 0.0062126
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 6.78
Output dim: 6, lower bound: -0.0062372, upper bound: 0.0062126
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 6.78
Output dim: 6, lower bound: -0.0062126, upper bound: 0.0062367
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 6.78
Output dim: 6, lower bound: -0.0062135, upper bound: 0.0062380
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 6.78
Output dim: 6, lower bound: -0.0062372, upper bound: 0.0062121
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 6.78
Output dim: 6, lower bound: -0.0062372, upper bound: 0.0062121
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 6.78
Output dim: 6, lower bound: -0.0062126, upper bound: 0.0062367
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 6.78
Output dim: 6, lower bound: -0.0062126, upper bound: 0.0062367

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 5.60 + 48.42 = 54.02 seconds
