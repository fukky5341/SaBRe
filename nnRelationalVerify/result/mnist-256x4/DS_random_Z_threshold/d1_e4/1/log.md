## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0302838


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107326, 0.0107326)
1: (0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176)
2: (0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544)
3: (-0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004)
4: (-0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596)
5: (0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762)
6: (-0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913)
7: (-0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638)
8: (0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195)
9: (-0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.05 + 2.79 = 3.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0356280, upper bound: 0.0356280

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0336867, upper bound: 0.0336867
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0336867, upper bound: 0.0336867
time: 1.73 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.45 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.45
Output dim: 8, lower bound: -0.0336867, upper bound: 0.0336867
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.45
Output dim: 8, lower bound: -0.0336867, upper bound: 0.0336867

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107077, 0.0107034
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0327689, upper bound: 0.0329135
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0329135, upper bound: 0.0327689
time: 1.58 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107034, 0.0107077
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313619, upper bound: 0.0313741
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313741, upper bound: 0.0313619
time: 1.41 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.35 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.35
Output dim: 8, lower bound: -0.0327689, upper bound: 0.0329135
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.35
Output dim: 8, lower bound: -0.0329135, upper bound: 0.0327689
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.35
Output dim: 8, lower bound: -0.0313619, upper bound: 0.0313741
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.35
Output dim: 8, lower bound: -0.0313741, upper bound: 0.0313619

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106670, 0.0106657
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325115, upper bound: 0.0326555
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325115, upper bound: 0.0326555
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106689, 0.0106626
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0326555, upper bound: 0.0325115
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0326555, upper bound: 0.0325115
time: 1.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106701, 0.0106770
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0283514, upper bound: 0.0283514
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0283514, upper bound: 0.0283514
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106718, 0.0106745
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305068, upper bound: 0.0305968
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306118, upper bound: 0.0304977
time: 1.56 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.79 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 8, lower bound: -0.0325115, upper bound: 0.0326555
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 8, lower bound: -0.0325115, upper bound: 0.0326555
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 8, lower bound: -0.0326555, upper bound: 0.0325115
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 8, lower bound: -0.0326555, upper bound: 0.0325115
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.79
Output dim: 8, lower bound: -0.0283514, upper bound: 0.0283514
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.79
Output dim: 8, lower bound: -0.0283514, upper bound: 0.0283514
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 8, lower bound: -0.0305068, upper bound: 0.0305968
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 8, lower bound: -0.0306118, upper bound: 0.0304977

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106450, 0.0106433
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313167, upper bound: 0.0316146
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0314681, upper bound: 0.0314604
time: 1.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106446, 0.0106456
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323459, upper bound: 0.0324840
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323409, upper bound: 0.0324865
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106471, 0.0106402
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0314833, upper bound: 0.0313553
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0314833, upper bound: 0.0313553
time: 2.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106465, 0.0106423
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0318907, upper bound: 0.0317534
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0318907, upper bound: 0.0317534
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106311, 0.0106357
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0290900, upper bound: 0.0291658
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0290900, upper bound: 0.0291658
time: 1.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106366, 0.0106338
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305819, upper bound: 0.0304701
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305819, upper bound: 0.0304721
time: 1.51 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 8, lower bound: -0.0313167, upper bound: 0.0316146
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 8, lower bound: -0.0314681, upper bound: 0.0314604
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 8, lower bound: -0.0323459, upper bound: 0.0324840
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 8, lower bound: -0.0323409, upper bound: 0.0324865
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 8, lower bound: -0.0314833, upper bound: 0.0313553
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 8, lower bound: -0.0314833, upper bound: 0.0313553
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 8, lower bound: -0.0318907, upper bound: 0.0317534
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 8, lower bound: -0.0318907, upper bound: 0.0317534
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.99
Output dim: 8, lower bound: -0.0290900, upper bound: 0.0291658
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.99
Output dim: 8, lower bound: -0.0290900, upper bound: 0.0291658
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 8, lower bound: -0.0305819, upper bound: 0.0304701
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 8, lower bound: -0.0305819, upper bound: 0.0304721

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105909, 0.0105902
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313167, upper bound: 0.0316146
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313167, upper bound: 0.0316114
time: 1.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105924, 0.0105892
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0314416, upper bound: 0.0314314
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0314414, upper bound: 0.0314278
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106438, 0.0106465
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311544, upper bound: 0.0314411
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0312990, upper bound: 0.0312890
time: 2.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106449, 0.0106448
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0319676, upper bound: 0.0321189
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0319749, upper bound: 0.0321102
time: 1.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106177, 0.0106238
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0284962, upper bound: 0.0284273
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0284962, upper bound: 0.0284273
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106252, 0.0106108
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299545, upper bound: 0.0299410
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300750, upper bound: 0.0298517
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106393, 0.0106337
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0295185, upper bound: 0.0294283
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0295185, upper bound: 0.0294283
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106379, 0.0106354
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0317615, upper bound: 0.0315034
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0316722, upper bound: 0.0316247
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106305, 0.0106260
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293358, upper bound: 0.0293718
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294721, upper bound: 0.0292271
time: 1.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106288, 0.0106308
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0274153, upper bound: 0.0273814
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0274153, upper bound: 0.0273814
time: 3.84 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 10.81 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0313167, upper bound: 0.0316146
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0313167, upper bound: 0.0316114
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0314416, upper bound: 0.0314314
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0314414, upper bound: 0.0314278
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0311544, upper bound: 0.0314411
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0312990, upper bound: 0.0312890
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0319676, upper bound: 0.0321189
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0319749, upper bound: 0.0321102
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0284962, upper bound: 0.0284273
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0284962, upper bound: 0.0284273
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0299545, upper bound: 0.0299410
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0300750, upper bound: 0.0298517
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0295185, upper bound: 0.0294283
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0295185, upper bound: 0.0294283
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0317615, upper bound: 0.0315034
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0316722, upper bound: 0.0316247
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0293358, upper bound: 0.0293718
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0294721, upper bound: 0.0292271
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0274153, upper bound: 0.0273814
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.81
Output dim: 8, lower bound: -0.0274153, upper bound: 0.0273814

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105762, 0.0105772
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298511, upper bound: 0.0302467
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299462, upper bound: 0.0301174
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105781, 0.0105754
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0312901, upper bound: 0.0315830
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0312885, upper bound: 0.0315775
time: 1.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105892, 0.0105817
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0312088, upper bound: 0.0312978
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313085, upper bound: 0.0312274
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105849, 0.0105840
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0312724, upper bound: 0.0312569
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0312703, upper bound: 0.0312710
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105899, 0.0105929
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298391, upper bound: 0.0300954
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298391, upper bound: 0.0300954
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105915, 0.0105926
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306814, upper bound: 0.0306882
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306814, upper bound: 0.0306882
time: 2.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106386, 0.0106388
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313497, upper bound: 0.0315204
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313727, upper bound: 0.0314937
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106373, 0.0106385
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0319476, upper bound: 0.0320827
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0319475, upper bound: 0.0320811
time: 1.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105790, 0.0105657
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311495, upper bound: 0.0309153
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311669, upper bound: 0.0308855
time: 1.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105682, 0.0105796
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309124, upper bound: 0.0308798
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309124, upper bound: 0.0308798
time: 1.71 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.32 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0298511, upper bound: 0.0302467
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0299462, upper bound: 0.0301174
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0312901, upper bound: 0.0315830
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0312885, upper bound: 0.0315775
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0312088, upper bound: 0.0312978
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0313085, upper bound: 0.0312274
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0312724, upper bound: 0.0312569
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0312703, upper bound: 0.0312710
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0298391, upper bound: 0.0300954
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0298391, upper bound: 0.0300954
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0306814, upper bound: 0.0306882
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0306814, upper bound: 0.0306882
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0313497, upper bound: 0.0315204
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0313727, upper bound: 0.0314937
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0319476, upper bound: 0.0320827
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0319475, upper bound: 0.0320811
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0311495, upper bound: 0.0309153
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0311669, upper bound: 0.0308855
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0309124, upper bound: 0.0308798
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 8, lower bound: -0.0309124, upper bound: 0.0308798

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105754, 0.0105681
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309216, upper bound: 0.0312139
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309266, upper bound: 0.0311979
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105708, 0.0105705
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298133, upper bound: 0.0302021
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299152, upper bound: 0.0300813
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104683, 0.0104685
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0212058, 0.0212083
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0312085, upper bound: 0.0312978
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0312088, upper bound: 0.0312978
time: 2.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104732, 0.0104607
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0212617, 0.0211198
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306933, upper bound: 0.0306504
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0307314, upper bound: 0.0306415
time: 1.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105844, 0.0105856
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0297915, upper bound: 0.0298912
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299052, upper bound: 0.0297925
time: 1.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105859, 0.0105836
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0310365, upper bound: 0.0311376
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311373, upper bound: 0.0310688
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105880, 0.0105893
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300289, upper bound: 0.0300693
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300610, upper bound: 0.0300640
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105880, 0.0105891
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305541, upper bound: 0.0304771
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304360, upper bound: 0.0305592
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106337, 0.0106338
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311124, upper bound: 0.0313841
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0312159, upper bound: 0.0313205
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106336, 0.0106338
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289429, upper bound: 0.0290408
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289429, upper bound: 0.0290397
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106334, 0.0106306
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313251, upper bound: 0.0314847
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313508, upper bound: 0.0314629
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106294, 0.0106327
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313249, upper bound: 0.0314832
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313509, upper bound: 0.0314603
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105740, 0.0105607
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0286731, upper bound: 0.0284999
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0286731, upper bound: 0.0284999
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105741, 0.0105609
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311662, upper bound: 0.0308853
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311669, upper bound: 0.0308855
time: 1.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105647, 0.0105760
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302740, upper bound: 0.0302648
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0302987, upper bound: 0.0302426
time: 1.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105649, 0.0105761
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0307316, upper bound: 0.0307527
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0307845, upper bound: 0.0306854
time: 1.87 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.87 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0309216, upper bound: 0.0312139
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0309266, upper bound: 0.0311979
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0298133, upper bound: 0.0302021
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0299152, upper bound: 0.0300813
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0312085, upper bound: 0.0312978
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0312088, upper bound: 0.0312978
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0306933, upper bound: 0.0306504
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0307314, upper bound: 0.0306415
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0297915, upper bound: 0.0298912
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0299052, upper bound: 0.0297925
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0310365, upper bound: 0.0311376
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0311373, upper bound: 0.0310688
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0300289, upper bound: 0.0300693
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0300610, upper bound: 0.0300640
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0305541, upper bound: 0.0304771
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0304360, upper bound: 0.0305592
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0311124, upper bound: 0.0313841
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0312159, upper bound: 0.0313205
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0289429, upper bound: 0.0290408
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0289429, upper bound: 0.0290397
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0313251, upper bound: 0.0314847
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0313508, upper bound: 0.0314629
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0313249, upper bound: 0.0314832
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0313509, upper bound: 0.0314603
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0286731, upper bound: 0.0284999
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0286731, upper bound: 0.0284999
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0311662, upper bound: 0.0308853
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0311669, upper bound: 0.0308855
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0302740, upper bound: 0.0302648
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0302987, upper bound: 0.0302426
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0307316, upper bound: 0.0307527
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 8, lower bound: -0.0307845, upper bound: 0.0306854

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105700, 0.0105635
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301895, upper bound: 0.0304713
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301895, upper bound: 0.0304713
time: 1.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105680, 0.0105626
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0250363, upper bound: 0.0251616
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0250363, upper bound: 0.0251616
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104534, 0.0104556
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0211248, 0.0211506
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305939, upper bound: 0.0307201
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306312, upper bound: 0.0307175
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104554, 0.0104536
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0211482, 0.0211272
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306250, upper bound: 0.0307130
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306250, upper bound: 0.0307130
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104674, 0.0104555
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0211921, 0.0210558
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299583, upper bound: 0.0299334
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299583, upper bound: 0.0299334
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104679, 0.0104558
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0211977, 0.0210593
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303548, upper bound: 0.0302742
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303642, upper bound: 0.0302701
time: 3.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104647, 0.0104686
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0213064, 0.0213500
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309051, upper bound: 0.0309224
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0307959, upper bound: 0.0310043
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104711, 0.0104624
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0213794, 0.0212797
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0252918, upper bound: 0.0252086
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0252918, upper bound: 0.0252086
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105312, 0.0105196
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0230896
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292813, upper bound: 0.0292266
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292813, upper bound: 0.0292266
time: 1.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105184, 0.0105322
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0230767, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300348, upper bound: 0.0301666
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300464, upper bound: 0.0301645
time: 1.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105161, 0.0105260
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0218947, 0.0220082
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0287030, upper bound: 0.0289067
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0287031, upper bound: 0.0289059
time: 1.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105208, 0.0105162
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0219487, 0.0218960
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311888, upper bound: 0.0312924
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311875, upper bound: 0.0312885
time: 1.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106280, 0.0106254
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289164, upper bound: 0.0290134
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289164, upper bound: 0.0290126
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106283, 0.0106255
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0312216, upper bound: 0.0312392
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311017, upper bound: 0.0313322
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106239, 0.0106275
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299618, upper bound: 0.0301338
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299618, upper bound: 0.0301338
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106242, 0.0106277
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0254329, upper bound: 0.0254658
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0254329, upper bound: 0.0254658
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105593, 0.0105482
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0286731, upper bound: 0.0284983
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0286731, upper bound: 0.0284983
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105613, 0.0105461
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303877, upper bound: 0.0301252
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303877, upper bound: 0.0301252
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105598, 0.0105716
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0302981, upper bound: 0.0302419
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0302987, upper bound: 0.0302426
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104455, 0.0104647
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0211147, 0.0213343
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305637, upper bound: 0.0305821
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305607, upper bound: 0.0305889
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104500, 0.0104567
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0211663, 0.0212427
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293125, upper bound: 0.0293082
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294145, upper bound: 0.0292152
time: 1.72 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.55 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0301895, upper bound: 0.0304713
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0301895, upper bound: 0.0304713
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0250363, upper bound: 0.0251616
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0250363, upper bound: 0.0251616
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0305939, upper bound: 0.0307201
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0306312, upper bound: 0.0307175
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0306250, upper bound: 0.0307130
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0306250, upper bound: 0.0307130
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0299583, upper bound: 0.0299334
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0299583, upper bound: 0.0299334
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0303548, upper bound: 0.0302742
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0303642, upper bound: 0.0302701
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0309051, upper bound: 0.0309224
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0307959, upper bound: 0.0310043
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0252918, upper bound: 0.0252086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0252918, upper bound: 0.0252086
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0292813, upper bound: 0.0292266
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0292813, upper bound: 0.0292266
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0300348, upper bound: 0.0301666
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0300464, upper bound: 0.0301645
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0287030, upper bound: 0.0289067
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0287031, upper bound: 0.0289059
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0311888, upper bound: 0.0312924
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0311875, upper bound: 0.0312885
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0289164, upper bound: 0.0290134
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0289164, upper bound: 0.0290126
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0312216, upper bound: 0.0312392
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0311017, upper bound: 0.0313322
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0299618, upper bound: 0.0301338
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0299618, upper bound: 0.0301338
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0254329, upper bound: 0.0254658
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0254329, upper bound: 0.0254658
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0286731, upper bound: 0.0284983
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0286731, upper bound: 0.0284983
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0303877, upper bound: 0.0301252
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0303877, upper bound: 0.0301252
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0302981, upper bound: 0.0302419
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0302987, upper bound: 0.0302426
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0305637, upper bound: 0.0305821
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0305607, upper bound: 0.0305889
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0293125, upper bound: 0.0293082
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.55
Output dim: 8, lower bound: -0.0294145, upper bound: 0.0292152

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105628, 0.0105547
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296119, upper bound: 0.0298982
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296188, upper bound: 0.0298629
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105611, 0.0105554
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294568, upper bound: 0.0296913
time: 2.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294568, upper bound: 0.0296913
time: 1.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104478, 0.0104504
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0210561, 0.0210860
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298716, upper bound: 0.0300026
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298716, upper bound: 0.0300026
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104481, 0.0104505
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0210602, 0.0210871
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0302525, upper bound: 0.0303450
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0302669, upper bound: 0.0303425
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104519, 0.0104508
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0211028, 0.0210895
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0281812, upper bound: 0.0282173
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0281812, upper bound: 0.0282146
time: 1.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104519, 0.0104501
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0211019, 0.0210819
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299705, upper bound: 0.0300977
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300069, upper bound: 0.0300940
time: 1.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104624, 0.0104508
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0212629, 0.0211306
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303533, upper bound: 0.0302742
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303548, upper bound: 0.0302742
time: 1.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104605, 0.0104502
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0212414, 0.0211245
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296858, upper bound: 0.0296427
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296858, upper bound: 0.0296427
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104088, 0.0103986
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0210224, 0.0209063
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305309, upper bound: 0.0305584
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305456, upper bound: 0.0305536
time: 1.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0103948, 0.0104095
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0208626, 0.0210307
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304190, upper bound: 0.0306415
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304365, upper bound: 0.0306400
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105143, 0.0105067
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0218895, 0.0218023
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296659, upper bound: 0.0298891
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0297708, upper bound: 0.0297620
time: 1.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105113, 0.0105079
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0218549, 0.0218165
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0287819, upper bound: 0.0287772
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0287819, upper bound: 0.0287742
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105705, 0.0105560
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298709, upper bound: 0.0299061
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298709, upper bound: 0.0299061
time: 1.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105587, 0.0105688
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299420, upper bound: 0.0301765
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299420, upper bound: 0.0301765
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105578, 0.0105426
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303603, upper bound: 0.0300993
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303617, upper bound: 0.0300991
time: 2.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105578, 0.0105426
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293090, upper bound: 0.0291303
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293964, upper bound: 0.0290593
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105450, 0.0105589
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302705, upper bound: 0.0302143
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302721, upper bound: 0.0302166
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105462, 0.0105567
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301357, upper bound: 0.0300702
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301279, upper bound: 0.0300834
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104443, 0.0104652
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0212264, 0.0214651
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0280262, upper bound: 0.0280936
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0280262, upper bound: 0.0280936
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104456, 0.0104635
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0212410, 0.0214459
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0290812, upper bound: 0.0292065
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0291900, upper bound: 0.0291245
time: 1.80 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 4.62 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0296119, upper bound: 0.0298982
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0296188, upper bound: 0.0298629
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0294568, upper bound: 0.0296913
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0294568, upper bound: 0.0296913
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0298716, upper bound: 0.0300026
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0298716, upper bound: 0.0300026
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0302525, upper bound: 0.0303450
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0302669, upper bound: 0.0303425
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0281812, upper bound: 0.0282173
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0281812, upper bound: 0.0282146
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0299705, upper bound: 0.0300977
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0300069, upper bound: 0.0300940
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0303533, upper bound: 0.0302742
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0303548, upper bound: 0.0302742
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0296858, upper bound: 0.0296427
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0296858, upper bound: 0.0296427
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0305309, upper bound: 0.0305584
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0305456, upper bound: 0.0305536
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0304190, upper bound: 0.0306415
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0304365, upper bound: 0.0306400
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0296659, upper bound: 0.0298891
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0297708, upper bound: 0.0297620
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0287819, upper bound: 0.0287772
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0287819, upper bound: 0.0287742
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0298709, upper bound: 0.0299061
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0298709, upper bound: 0.0299061
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0299420, upper bound: 0.0301765
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0299420, upper bound: 0.0301765
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0303603, upper bound: 0.0300993
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0303617, upper bound: 0.0300991
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0293090, upper bound: 0.0291303
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0293964, upper bound: 0.0290593
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0302705, upper bound: 0.0302143
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0302721, upper bound: 0.0302166
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0301357, upper bound: 0.0300702
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0301279, upper bound: 0.0300834
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0280262, upper bound: 0.0280936
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0280262, upper bound: 0.0280936
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0290812, upper bound: 0.0292065
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.62
Output dim: 8, lower bound: -0.0291900, upper bound: 0.0291245

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104442, 0.0104478
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0211463, 0.0211873
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0240386, upper bound: 0.0240448
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0240386, upper bound: 0.0240448
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104425, 0.0104465
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0211276, 0.0211732
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289030, upper bound: 0.0289918
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289030, upper bound: 0.0289918
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104491, 0.0104394
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0212022, 0.0210919
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0278584, upper bound: 0.0277401
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0278584, upper bound: 0.0277397
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104512, 0.0104375
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0212262, 0.0210699
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302300, upper bound: 0.0300687
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301108, upper bound: 0.0301467
time: 1.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104033, 0.0103933
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0209990, 0.0208845
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299203, upper bound: 0.0299462
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299203, upper bound: 0.0299462
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0104015, 0.0103932
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0209775, 0.0208829
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0291703, upper bound: 0.0292212
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0291703, upper bound: 0.0292212
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0103893, 0.0104034
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0208392, 0.0209994
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0280897, upper bound: 0.0282724
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0280897, upper bound: 0.0282699
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0103890, 0.0104041
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0208356, 0.0210073
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304365, upper bound: 0.0306392
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304363, upper bound: 0.0306400
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105544, 0.0105354
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299698, upper bound: 0.0297158
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299732, upper bound: 0.0297090
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105505, 0.0105386
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299708, upper bound: 0.0297155
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299748, upper bound: 0.0297091
time: 1.66 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 4.24 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0240386, upper bound: 0.0240448
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0240386, upper bound: 0.0240448
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0289030, upper bound: 0.0289918
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0289030, upper bound: 0.0289918
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0278584, upper bound: 0.0277401
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0278584, upper bound: 0.0277397
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0302300, upper bound: 0.0300687
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0301108, upper bound: 0.0301467
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0299203, upper bound: 0.0299462
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0299203, upper bound: 0.0299462
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0291703, upper bound: 0.0292212
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0291703, upper bound: 0.0292212
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0280897, upper bound: 0.0282724
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0280897, upper bound: 0.0282699
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0304365, upper bound: 0.0306392
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0304363, upper bound: 0.0306400
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0299698, upper bound: 0.0297158
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0299732, upper bound: 0.0297090
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0299708, upper bound: 0.0297155
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 8, lower bound: -0.0299748, upper bound: 0.0297091

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0103758, 0.0103930
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0208219, 0.0210182
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298163, upper bound: 0.0300603
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298649, upper bound: 0.0300574
time: 1.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0103785, 0.0103908
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0208527, 0.0209937
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0290774, upper bound: 0.0292901
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0290774, upper bound: 0.0292901
time: 1.50 seconds

## Summary of splitting (split count: 10)
- Time for DS candidates: 3.94 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.94
Output dim: 8, lower bound: -0.0298163, upper bound: 0.0300603
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.94
Output dim: 8, lower bound: -0.0298649, upper bound: 0.0300574
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.94
Output dim: 8, lower bound: -0.0290774, upper bound: 0.0292901
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.94
Output dim: 8, lower bound: -0.0290774, upper bound: 0.0292901

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.84 + 489.45 = 493.29 seconds
