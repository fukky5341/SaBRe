## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 3.5844923581200003


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910)
1: (-6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244)
2: (-3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980)
3: (-4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017)
4: (-2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.95 + 1.50 = 2.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -3.5852094, upper bound: 3.5852094

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850997, upper bound: 3.5850997
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850997, upper bound: 3.5851529
time: 0.56 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.07 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.07
Output dim: 4, lower bound: -3.5850997, upper bound: 3.5850997
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.07
Output dim: 4, lower bound: -3.5850997, upper bound: 3.5851529

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849262, upper bound: 3.5849262
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849262, upper bound: 3.5849288
time: 0.50 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850992, upper bound: 3.5851355
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850990, upper bound: 3.5851516
time: 0.48 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.91 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.91
Output dim: 4, lower bound: -3.5849262, upper bound: 3.5849262
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.91
Output dim: 4, lower bound: -3.5849262, upper bound: 3.5849288
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.91
Output dim: 4, lower bound: -3.5850992, upper bound: 3.5851355
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.91
Output dim: 4, lower bound: -3.5850990, upper bound: 3.5851516

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848441, upper bound: 3.5849110
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849672, upper bound: 3.5848441
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849858, upper bound: 3.5849274
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848958, upper bound: 3.5849274
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849274, upper bound: 3.5848958
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849246, upper bound: 3.5849701
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847006, upper bound: 3.5847954
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847538, upper bound: 3.5847840
time: 0.52 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.90 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.90
Output dim: 4, lower bound: -3.5848441, upper bound: 3.5849110
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.90
Output dim: 4, lower bound: -3.5849672, upper bound: 3.5848441
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.90
Output dim: 4, lower bound: -3.5849858, upper bound: 3.5849274
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.90
Output dim: 4, lower bound: -3.5848958, upper bound: 3.5849274
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.90
Output dim: 4, lower bound: -3.5849274, upper bound: 3.5848958
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.90
Output dim: 4, lower bound: -3.5849246, upper bound: 3.5849701
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.90
Output dim: 4, lower bound: -3.5847006, upper bound: 3.5847954
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.90
Output dim: 4, lower bound: -3.5847538, upper bound: 3.5847840

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848441, upper bound: 3.5848441
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848441, upper bound: 3.5849110
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849000, upper bound: 3.5847924
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849851, upper bound: 3.5849272
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849858, upper bound: 3.5848661
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848173, upper bound: 3.5848607
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848463, upper bound: 3.5848604
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845051
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845522, upper bound: 3.5845423
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848586, upper bound: 3.5849066
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848173, upper bound: 3.5848173
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846362, upper bound: 3.5847214
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846362, upper bound: 3.5846718
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845522, upper bound: 3.5845818
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845969
time: 0.61 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5848441, upper bound: 3.5848441
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5848441, upper bound: 3.5849110
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5849000, upper bound: 3.5847924
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5849851, upper bound: 3.5849272
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5849858, upper bound: 3.5848661
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5848173, upper bound: 3.5848607
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5848463, upper bound: 3.5848604
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845051
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5845522, upper bound: 3.5845423
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5848586, upper bound: 3.5849066
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5848173, upper bound: 3.5848173
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5846362, upper bound: 3.5847214
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5846362, upper bound: 3.5846718
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5845522, upper bound: 3.5845818
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845969

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846673, upper bound: 3.5846673
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846673, upper bound: 3.5846673
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844849, upper bound: 3.5845184
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844849, upper bound: 3.5844849
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848891, upper bound: 3.5847924
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848661, upper bound: 3.5849272
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848661, upper bound: 3.5848661
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849858, upper bound: 3.5848661
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849853, upper bound: 3.5848661
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5844702
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5844289
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844638, upper bound: 3.5844654
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5844289
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844539, upper bound: 3.5844539
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844539, upper bound: 3.5844539
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845051
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845522, upper bound: 3.5845423
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848466
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847892
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846021, upper bound: 3.5847174
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846021, upper bound: 3.5846917
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844873, upper bound: 3.5845527
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844873, upper bound: 3.5844903
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844638, upper bound: 3.5844990
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5844289
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845969
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845051
time: 0.48 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5846673, upper bound: 3.5846673
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5846673, upper bound: 3.5846673
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5844849, upper bound: 3.5845184
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5844849, upper bound: 3.5844849
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5848891, upper bound: 3.5847924
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5848661, upper bound: 3.5849272
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5848661, upper bound: 3.5848661
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5849858, upper bound: 3.5848661
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5849853, upper bound: 3.5848661
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5844702
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5844289
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5844638, upper bound: 3.5844654
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5844289
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5844539, upper bound: 3.5844539
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5844539, upper bound: 3.5844539
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845051
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5845522, upper bound: 3.5845423
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848466
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847892
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5846021, upper bound: 3.5847174
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5846021, upper bound: 3.5846917
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5844873, upper bound: 3.5845527
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5844873, upper bound: 3.5844903
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5844638, upper bound: 3.5844990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5844289
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845969
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.04
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845051

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843462, upper bound: 3.5843462
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844360, upper bound: 3.5843462
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843485, upper bound: 3.5843462
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843462, upper bound: 3.5843462
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844849, upper bound: 3.5845184
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844849, upper bound: 3.5844849
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848780, upper bound: 3.5847924
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849190, upper bound: 3.5848606
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849187, upper bound: 3.5848555
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847939, upper bound: 3.5847939
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847939, upper bound: 3.5847939
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845051
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845051
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847939, upper bound: 3.5847939
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847939, upper bound: 3.5847939
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844849, upper bound: 3.5844849
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844849, upper bound: 3.5844849
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845423
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845522, upper bound: 3.5845051
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848466
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848132, upper bound: 3.5847892
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5845096
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5845287
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5846733
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5846811
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844795, upper bound: 3.5845527
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844795, upper bound: 3.5845472
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5844990
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844638, upper bound: 3.5844662
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5845129
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5844289
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843462, upper bound: 3.5843462
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843462, upper bound: 3.5843462
time: 0.47 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.84 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5843462, upper bound: 3.5843462
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5844360, upper bound: 3.5843462
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5843485, upper bound: 3.5843462
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5843462, upper bound: 3.5843462
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5844849, upper bound: 3.5845184
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5844849, upper bound: 3.5844849
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5848780, upper bound: 3.5847924
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5849190, upper bound: 3.5848606
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5849187, upper bound: 3.5848555
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847939, upper bound: 3.5847939
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847939, upper bound: 3.5847939
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845051
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845051
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847939, upper bound: 3.5847939
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847939, upper bound: 3.5847939
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5844849, upper bound: 3.5844849
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5844849, upper bound: 3.5844849
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5845051, upper bound: 3.5845423
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5845522, upper bound: 3.5845051
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848466
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5848132, upper bound: 3.5847892
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5845096
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5845287
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5846733
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5846811
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5844795, upper bound: 3.5845527
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5844795, upper bound: 3.5845472
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5844990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5844638, upper bound: 3.5844662
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5845129
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5844289
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5843462, upper bound: 3.5843462
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.84
Output dim: 4, lower bound: -3.5843462, upper bound: 3.5843462

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844180
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844450
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846065, upper bound: 3.5846065
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846065, upper bound: 3.5846065
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848490, upper bound: 3.5847924
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845603, upper bound: 3.5845603
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846839, upper bound: 3.5845603
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846045, upper bound: 3.5846045
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846045, upper bound: 3.5846045
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845085, upper bound: 3.5844180
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844180
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844180
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844180
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846103, upper bound: 3.5846633
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846103, upper bound: 3.5846068
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848081
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846170, upper bound: 3.5846170
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846170, upper bound: 3.5846170
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844539, upper bound: 3.5844539
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844539, upper bound: 3.5844539
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5844289
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844691, upper bound: 3.5844289
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847048, upper bound: 3.5846170
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846170, upper bound: 3.5846170
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846170, upper bound: 3.5846170
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846170, upper bound: 3.5846170
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844539, upper bound: 3.5844539
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844539, upper bound: 3.5844860
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843462, upper bound: 3.5843462
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844223, upper bound: 3.5843462
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5846642
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5846473
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848132, upper bound: 3.5847892
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5844882
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5844803
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5843779
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5843575
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5846733
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5845912
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5846799
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5846811
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5845426
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5845415
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844795, upper bound: 3.5844923
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844795, upper bound: 3.5845472
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844189
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844837
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5845085
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844956
time: 0.50 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.16 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844180
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844450
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5846065, upper bound: 3.5846065
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5846065, upper bound: 3.5846065
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5848490, upper bound: 3.5847924
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5845603, upper bound: 3.5845603
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5846839, upper bound: 3.5845603
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5846045, upper bound: 3.5846045
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5846045, upper bound: 3.5846045
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5845085, upper bound: 3.5844180
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844180
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844180
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844180
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847924, upper bound: 3.5847924
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5846103, upper bound: 3.5846633
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5846103, upper bound: 3.5846068
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848081
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5846170, upper bound: 3.5846170
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5846170, upper bound: 3.5846170
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844539, upper bound: 3.5844539
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844539, upper bound: 3.5844539
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844289, upper bound: 3.5844289
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844691, upper bound: 3.5844289
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847048, upper bound: 3.5846170
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5846170, upper bound: 3.5846170
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5846170, upper bound: 3.5846170
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5846170, upper bound: 3.5846170
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844539, upper bound: 3.5844539
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844539, upper bound: 3.5844860
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5843462, upper bound: 3.5843462
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844223, upper bound: 3.5843462
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5846642
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5846473
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5848132, upper bound: 3.5847892
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5844882
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5844803
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5843779
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5843575
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5846733
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5845912
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5846799
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5846811
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5845426
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5845415
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844795, upper bound: 3.5844923
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844795, upper bound: 3.5845472
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844189
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844837
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5845085
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844956

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845603, upper bound: 3.5845603
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845603, upper bound: 3.5845603
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846045, upper bound: 3.5846045
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846045, upper bound: 3.5846045
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846045, upper bound: 3.5846045
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846607, upper bound: 3.5846045
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845603, upper bound: 3.5845603
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845603, upper bound: 3.5845603
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845603, upper bound: 3.5845603
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845603, upper bound: 3.5845603
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846045, upper bound: 3.5846045
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846045, upper bound: 3.5846045
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843954, upper bound: 3.5843951
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844798, upper bound: 3.5843951
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844180
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844180
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845566, upper bound: 3.5846315
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5842603
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5842603
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845888, upper bound: 3.5846251
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845888, upper bound: 3.5845538
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843235, upper bound: 3.5843235
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843235, upper bound: 3.5843235
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843369, upper bound: 3.5843235
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844087, upper bound: 3.5843235
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843254, upper bound: 3.5843235
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843235, upper bound: 3.5843235
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843251, upper bound: 3.5843235
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844367, upper bound: 3.5843235
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5846642
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5846473
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5846164
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5844392
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5846733
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5845912
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5844699
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5844699
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5846799
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5845912
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5844760
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5844583
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5845426
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5844699
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5845414
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5845415
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5844699
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5845398
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5844798
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843954
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844956
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844847
time: 0.67 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.47 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845603, upper bound: 3.5845603
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845603, upper bound: 3.5845603
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5846045, upper bound: 3.5846045
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5846045, upper bound: 3.5846045
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5846045, upper bound: 3.5846045
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5846607, upper bound: 3.5846045
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845603, upper bound: 3.5845603
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845603, upper bound: 3.5845603
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845603, upper bound: 3.5845603
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845603, upper bound: 3.5845603
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5846045, upper bound: 3.5846045
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5846045, upper bound: 3.5846045
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843954, upper bound: 3.5843951
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5844798, upper bound: 3.5843951
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844180
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844180
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845566, upper bound: 3.5846315
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5842603
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5842603
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845888, upper bound: 3.5846251
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845888, upper bound: 3.5845538
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843235, upper bound: 3.5843235
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843235, upper bound: 3.5843235
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843369, upper bound: 3.5843235
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5844087, upper bound: 3.5843235
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843254, upper bound: 3.5843235
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843235, upper bound: 3.5843235
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843251, upper bound: 3.5843235
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5844367, upper bound: 3.5843235
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5846642
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5846473
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5846164
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5844392
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5846733
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5845912
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5844699
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5844699
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5846799
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5845912, upper bound: 3.5845912
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5844760
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5844583
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5845426
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5844699
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5845414
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5845415
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5844699
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5844699, upper bound: 3.5845398
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5844798
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843954
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844956
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 4, lower bound: -3.5844180, upper bound: 3.5844847

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842871, upper bound: 3.5842871
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842871, upper bound: 3.5842871
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842871, upper bound: 3.5842871
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842871, upper bound: 3.5842871
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5842603
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5842603
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5842603
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5842603
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5842603
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5842603
time: 2.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5842603
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5842603
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842871, upper bound: 3.5842870
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842871, upper bound: 3.5842871
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842871, upper bound: 3.5842870
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842871, upper bound: 3.5842870
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842871, upper bound: 3.5842871
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844072, upper bound: 3.5842871
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5842603
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842603, upper bound: 3.5842603
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846473, upper bound: 3.5845538
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846543, upper bound: 3.5845538
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5843181
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842392, upper bound: 3.5842387
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845888, upper bound: 3.5846251
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842683, upper bound: 3.5842387
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845833, upper bound: 3.5845538
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5846543
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5843318
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5843271
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5846164
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845926
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842388
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842387, upper bound: 3.5842387
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843951, upper bound: 3.5843951
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.26 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.45 + 417.56 = 420.01 seconds
