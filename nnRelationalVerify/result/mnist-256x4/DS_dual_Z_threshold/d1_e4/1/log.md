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
execution time: IAR + RelationalAnalysis = 1.88 + 2.99 = 4.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0356280, upper bound: 0.0356280

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0334691, upper bound: 0.0334691
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0334691, upper bound: 0.0334691
time: 2.22 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.35 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.35
Output dim: 8, lower bound: -0.0334691, upper bound: 0.0334691
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.35
Output dim: 8, lower bound: -0.0334691, upper bound: 0.0334691

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107032, 0.0107072
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0333050, upper bound: 0.0332985
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0332985, upper bound: 0.0333050
time: 1.94 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107072, 0.0107032
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0333050, upper bound: 0.0332985
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0332985, upper bound: 0.0333050
time: 1.66 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.07 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.07
Output dim: 8, lower bound: -0.0333050, upper bound: 0.0332985
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.07
Output dim: 8, lower bound: -0.0332985, upper bound: 0.0333050
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.07
Output dim: 8, lower bound: -0.0333050, upper bound: 0.0332985
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.07
Output dim: 8, lower bound: -0.0332985, upper bound: 0.0333050

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107023, 0.0107077
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107034, 0.0107063
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107063, 0.0107034
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
time: 1.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107077, 0.0107023
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816
time: 1.83 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.12 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.12
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.12
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.12
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.12
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.12
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.12
Output dim: 8, lower bound: -0.0325816, upper bound: 0.0325739
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.12
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.12
Output dim: 8, lower bound: -0.0325739, upper bound: 0.0325816

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106775, 0.0106879
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
time: 1.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106732, 0.0106829
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106786, 0.0106861
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106751, 0.0106815
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106815, 0.0106751
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
time: 2.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106861, 0.0106786
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106829, 0.0106732
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545
time: 1.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106879, 0.0106775
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545
time: 1.77 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.69 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0324545, upper bound: 0.0323343
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0323390, upper bound: 0.0324464
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0324464, upper bound: 0.0323390
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.69
Output dim: 8, lower bound: -0.0323343, upper bound: 0.0324545

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106200, 0.0106178
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
time: 1.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106075, 0.0106275
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
time: 1.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106163, 0.0106129
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106031, 0.0106233
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106213, 0.0106160
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106086, 0.0106257
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823
time: 3.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106183, 0.0106114
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106050, 0.0106224
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823
time: 3.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106224, 0.0106050
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
time: 1.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106114, 0.0106183
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106257, 0.0106086
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106160, 0.0106213
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106233, 0.0106031
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106128, 0.0106163
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823
time: 3.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106275, 0.0106075
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0106178, 0.0106200
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823
time: 4.30 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 8.08 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309823, upper bound: 0.0309917
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0311072, upper bound: 0.0308624
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0308687, upper bound: 0.0311067
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309923, upper bound: 0.0309704
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309704, upper bound: 0.0309923
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0311067, upper bound: 0.0308687
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0308624, upper bound: 0.0311072
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 8, lower bound: -0.0309917, upper bound: 0.0309823

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105672, 0.0105628
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105612, 0.0105651
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105547, 0.0105730
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105483, 0.0105748
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 2.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105635, 0.0105575
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 2.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105599, 0.0105601
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105504, 0.0105669
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105466, 0.0105705
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 1.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105686, 0.0105602
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105631, 0.0105633
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105558, 0.0105707
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105504, 0.0105730
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105655, 0.0105555
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105624, 0.0105586
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105522, 0.0105653
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105493, 0.0105696
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105696, 0.0105493
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105653, 0.0105522
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105586, 0.0105624
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105555, 0.0105655
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 2.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105730, 0.0105504
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
time: 1.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105707, 0.0105558
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105633, 0.0105631
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
time: 1.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105602, 0.0105686
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
time: 2.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105705, 0.0105466
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105669, 0.0105504
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105601, 0.0105599
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105575, 0.0105635
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 2.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105748, 0.0105483
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105730, 0.0105547
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105651, 0.0105612
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105628, 0.0105672
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
time: 1.82 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 5.21 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302498, upper bound: 0.0302369
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303540, upper bound: 0.0301312
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301383, upper bound: 0.0303466
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302376, upper bound: 0.0302336
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302336, upper bound: 0.0302376
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0303466, upper bound: 0.0301383
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0301312, upper bound: 0.0303540
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 8, lower bound: -0.0302369, upper bound: 0.0302498

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105577, 0.0105617
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 1.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105579, 0.0105615
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105512, 0.0105696
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105512, 0.0105695
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105563, 0.0105566
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 1.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105564, 0.0105566
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 1.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105468, 0.0105636
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105470, 0.0105634
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105596, 0.0105599
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105598, 0.0105598
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105523, 0.0105674
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105523, 0.0105672
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105589, 0.0105552
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105590, 0.0105551
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105487, 0.0105619
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105489, 0.0105618
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105618, 0.0105489
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105619, 0.0105487
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 1.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105551, 0.0105590
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105552, 0.0105589
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105672, 0.0105523
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 1.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105674, 0.0105523
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
time: 1.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105598, 0.0105598
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105599, 0.0105596
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
time: 1.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105634, 0.0105470
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105636, 0.0105468
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105566, 0.0105564
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105566, 0.0105563
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105695, 0.0105512
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105696, 0.0105512
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105615, 0.0105579
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0105617, 0.0105577
1: 0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176
2: 0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544
3: -0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004
4: -0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596
5: 0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762
6: -0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913
7: -0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638
8: 0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195
9: -0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
time: 1.74 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.29 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296711, upper bound: 0.0294744
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296890, upper bound: 0.0294513
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294634, upper bound: 0.0296779
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294843, upper bound: 0.0296596
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296596, upper bound: 0.0294843
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0296779, upper bound: 0.0294634
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294513, upper bound: 0.0296890
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 8, lower bound: -0.0294744, upper bound: 0.0296711

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.87 + 499.31 = 504.18 seconds
