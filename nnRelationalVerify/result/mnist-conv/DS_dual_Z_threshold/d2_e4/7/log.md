## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.6321674314


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5920982, 1.5920992)
1: (-12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7782269, 1.7782259)
2: (-8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9221311, 1.9221311)
3: (-10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9307184, 1.9307184)
4: (-4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6834741, 1.6834741)
5: (-2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6719646, 1.6719651)
6: (9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.1016169, 1.1016171)
7: (-21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7424874, 1.7424872)
8: (-2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3420033, 1.3420031)
9: (-13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.5054874, 1.5054879)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.37 + 49.77 = 73.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.6334342, upper bound: 0.6334355

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6113
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 6213
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6113

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334325, upper bound: 0.6302066
time: 9.92 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6302069, upper bound: 0.6334339
time: 8.19 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 18.38 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 18.38
Output dim: 6, lower bound: -0.6334325, upper bound: 0.6302066
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 18.38
Output dim: 6, lower bound: -0.6302069, upper bound: 0.6334339

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5918097, 1.5917091
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7773457, 1.7775722
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9179344, 1.9164810
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9266405, 1.9276938
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6818476, 1.6812830
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6707630, 1.6703453
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0940881, 1.0914645
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7399564, 1.7406104
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3415294, 1.3416514
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4988632, 1.5005717

Time for backsubstitution: 21.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 6213
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 6111

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334319, upper bound: 0.6298229
time: 9.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6330478, upper bound: 0.6302061
time: 7.04 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5917087, 1.5918097
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7775726, 1.7773457
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9164810, 1.9179339
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9276943, 1.9266405
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6812830, 1.6818471
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6703444, 1.6707635
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0914645, 1.0940883
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7406101, 1.7399566
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3416514, 1.3415291
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.5005713, 1.4988627

Time for backsubstitution: 21.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 6213
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6111

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6302063, upper bound: 0.6330492
time: 5.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6298229, upper bound: 0.6334321
time: 8.63 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 36.11 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 36.11
Output dim: 6, lower bound: -0.6334319, upper bound: 0.6298229
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 36.11
Output dim: 6, lower bound: -0.6330478, upper bound: 0.6302061
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 36.11
Output dim: 6, lower bound: -0.6302063, upper bound: 0.6330492
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 36.11
Output dim: 6, lower bound: -0.6298229, upper bound: 0.6334321

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5911107, 1.5913701
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7768693, 1.7765956
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9166193, 1.9137859
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9262338, 1.9268618
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6805038, 1.6785340
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6657515, 1.6679044
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0930166, 1.0909400
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7382903, 1.7397971
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3413148, 1.3412120
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4928579, 1.4976482

Time for backsubstitution: 22.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6213
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 6213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334309, upper bound: 0.6290665
time: 8.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6326760, upper bound: 0.6298219
time: 6.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5914702, 1.5910106
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7763696, 1.7770953
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9152393, 1.9151664
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9258084, 1.9272866
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6790981, 1.6799393
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6683226, 1.6653323
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0935631, 1.0903933
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7391429, 1.7389441
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3410897, 1.3414369
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4959402, 1.4945669

Time for backsubstitution: 22.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6213
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 6213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6330468, upper bound: 0.6294493
time: 6.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6322924, upper bound: 0.6302048
time: 8.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5910106, 1.5914702
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7770953, 1.7763691
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9151669, 1.9152393
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9272866, 1.9258084
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6799393, 1.6790981
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6653318, 1.6683226
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0903931, 1.0935636
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7389436, 1.7391434
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3414369, 1.3410897
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4945669, 1.4959397

Time for backsubstitution: 22.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6213
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 6213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6302053, upper bound: 0.6322920
time: 9.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6294496, upper bound: 0.6330467
time: 8.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5913701, 1.5911107
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7765956, 1.7768688
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9137859, 1.9166198
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9268613, 1.9262333
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6785336, 1.6805034
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6679029, 1.6657510
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0909395, 1.0930171
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7397971, 1.7382903
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3412123, 1.3413146
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4976482, 1.4928579

Time for backsubstitution: 22.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6213
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 6213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6298219, upper bound: 0.6326760
time: 7.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6290665, upper bound: 0.6334323
time: 4.35 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 34.62 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.62
Output dim: 6, lower bound: -0.6334309, upper bound: 0.6290665
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.62
Output dim: 6, lower bound: -0.6326760, upper bound: 0.6298219
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.62
Output dim: 6, lower bound: -0.6330468, upper bound: 0.6294493
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.62
Output dim: 6, lower bound: -0.6322924, upper bound: 0.6302048
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.62
Output dim: 6, lower bound: -0.6302053, upper bound: 0.6322920
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.62
Output dim: 6, lower bound: -0.6294496, upper bound: 0.6330467
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.62
Output dim: 6, lower bound: -0.6298219, upper bound: 0.6326760
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.62
Output dim: 6, lower bound: -0.6290665, upper bound: 0.6334323

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5878015, 1.5866561
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7552710, 1.7576981
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.8991599, 1.8938298
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9308028, 1.9309206
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6537027, 1.6550803
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6525822, 1.6563783
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0906348, 1.0868626
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7292476, 1.7359226
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3473506, 1.3440099
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4770169, 1.4837823

Time for backsubstitution: 22.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334276, upper bound: 0.6253607
time: 10.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6297224, upper bound: 0.6290648
time: 4.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5863967, 1.5880609
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7579718, 1.7549968
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.8966632, 1.8963261
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9302917, 1.9314308
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6570492, 1.6517339
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6542244, 1.6547360
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0889397, 1.0885577
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7344155, 1.7307544
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3441124, 1.3472481
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4789920, 1.4818077

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6326729, upper bound: 0.6261148
time: 4.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6289706, upper bound: 0.6298189
time: 11.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5881619, 1.5862966
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7547703, 1.7581983
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.8977790, 1.8952098
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9303775, 1.9313455
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6522980, 1.6564856
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6551533, 1.6538067
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0911818, 1.0863159
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7301006, 1.7350695
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3471260, 1.3442345
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4800982, 1.4807005

Time for backsubstitution: 22.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 6184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6330435, upper bound: 0.6257441
time: 8.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6293376, upper bound: 0.6294464
time: 8.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5867562, 1.5877013
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7574720, 1.7554965
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.8952823, 1.8977065
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9298673, 1.9318562
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6556444, 1.6531391
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6567974, 1.6521640
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0894866, 1.0880113
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7352691, 1.7299013
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3438878, 1.3474729
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4820733, 1.4787259

Time for backsubstitution: 22.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 6184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6322893, upper bound: 0.6264971
time: 9.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6285866, upper bound: 0.6302019
time: 7.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5877013, 1.5867562
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7554960, 1.7574716
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.8977065, 1.8952827
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9318557, 1.9298673
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6531391, 1.6556444
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6521645, 1.6567969
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0880113, 1.0894861
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7299013, 1.7352688
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3474727, 1.3438876
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4787259, 1.4820738

Time for backsubstitution: 22.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 6184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6302020, upper bound: 0.6285880
time: 5.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6264971, upper bound: 0.6322906
time: 6.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5862966, 1.5881610
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7581987, 1.7547703
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.8952098, 1.8977790
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9313455, 1.9303780
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6564856, 1.6522985
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6538067, 1.6551542
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0863161, 1.0911815
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7350698, 1.7301006
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3442349, 1.3471260
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4807010, 1.4800987

Time for backsubstitution: 22.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 6184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6294465, upper bound: 0.6293378
time: 7.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6257442, upper bound: 0.6330434
time: 7.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5880609, 1.5863967
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7549963, 1.7579718
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.8963256, 1.8966632
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9314313, 1.9302921
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6517344, 1.6570497
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6547356, 1.6542249
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0885577, 1.0889397
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7307544, 1.7344158
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3472486, 1.3441122
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4818072, 1.4789920

Time for backsubstitution: 22.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6298186, upper bound: 0.6289719
time: 7.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6261134, upper bound: 0.6326726
time: 7.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5866551, 1.5878015
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7576990, 1.7552700
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.8938289, 1.8991594
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9309211, 1.9308028
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6550798, 1.6537037
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6563778, 1.6525826
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0868626, 1.0906351
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7359223, 1.7292476
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3440099, 1.3473506
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4837823, 1.4770174

Time for backsubstitution: 22.52 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 73.14 + 527.88 = 601.02 seconds
