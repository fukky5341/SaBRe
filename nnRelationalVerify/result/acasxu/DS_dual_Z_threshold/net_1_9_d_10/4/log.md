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
execution time: IAR + RelationalAnalysis = 1.08 + 1.49 = 2.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -3.5852094, upper bound: 3.5852094

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851962, upper bound: 3.5851896
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851896, upper bound: 3.5851962
time: 0.76 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.44 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 4, lower bound: -3.5851962, upper bound: 3.5851896
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 4, lower bound: -3.5851896, upper bound: 3.5851962

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851917, upper bound: 3.5851896
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851962, upper bound: 3.5851611
time: 0.67 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851611, upper bound: 3.5851962
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851611, upper bound: 3.5851917
time: 0.44 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.03 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 4, lower bound: -3.5851917, upper bound: 3.5851896
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 4, lower bound: -3.5851962, upper bound: 3.5851611
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 4, lower bound: -3.5851611, upper bound: 3.5851962
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 4, lower bound: -3.5851611, upper bound: 3.5851917

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851424, upper bound: 3.5851377
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851425, upper bound: 3.5851158
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851423, upper bound: 3.5851136
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851428, upper bound: 3.5850633
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850633, upper bound: 3.5851428
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850633, upper bound: 3.5851423
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850633, upper bound: 3.5851425
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850633, upper bound: 3.5851424
time: 0.67 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.44 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 4, lower bound: -3.5851424, upper bound: 3.5851377
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 4, lower bound: -3.5851425, upper bound: 3.5851158
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 4, lower bound: -3.5851423, upper bound: 3.5851136
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 4, lower bound: -3.5851428, upper bound: 3.5850633
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 4, lower bound: -3.5850633, upper bound: 3.5851428
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 4, lower bound: -3.5850633, upper bound: 3.5851423
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 4, lower bound: -3.5850633, upper bound: 3.5851425
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 4, lower bound: -3.5850633, upper bound: 3.5851424

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850555, upper bound: 3.5850697
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850663, upper bound: 3.5850349
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850555, upper bound: 3.5850501
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850628, upper bound: 3.5849691
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849691, upper bound: 3.5849701
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850745, upper bound: 3.5850179
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849691, upper bound: 3.5849695
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850749, upper bound: 3.5849693
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849693, upper bound: 3.5850749
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849695, upper bound: 3.5850515
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849693, upper bound: 3.5850745
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849701, upper bound: 3.5849691
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849691, upper bound: 3.5850628
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849695, upper bound: 3.5850555
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849691, upper bound: 3.5850628
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849695, upper bound: 3.5850555
time: 0.47 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.10 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5850555, upper bound: 3.5850697
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5850663, upper bound: 3.5850349
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5850555, upper bound: 3.5850501
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5850628, upper bound: 3.5849691
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5849691, upper bound: 3.5849701
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5850745, upper bound: 3.5850179
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5849691, upper bound: 3.5849695
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5850749, upper bound: 3.5849693
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5849693, upper bound: 3.5850749
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5849695, upper bound: 3.5850515
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5849693, upper bound: 3.5850745
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5849701, upper bound: 3.5849691
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5849691, upper bound: 3.5850628
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5849695, upper bound: 3.5850555
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5849691, upper bound: 3.5850628
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 4, lower bound: -3.5849695, upper bound: 3.5850555

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5848704
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849331, upper bound: 3.5849428
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5848700
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849401, upper bound: 3.5849145
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5848704
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849331, upper bound: 3.5849282
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848714, upper bound: 3.5848700
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848706, upper bound: 3.5848700
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5848706
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5848709
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849415, upper bound: 3.5848845
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849424, upper bound: 3.5848991
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849247, upper bound: 3.5848703
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849322, upper bound: 3.5848702
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849437, upper bound: 3.5848702
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849436, upper bound: 3.5848700
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5849436
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848702, upper bound: 3.5849437
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848702, upper bound: 3.5849322
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848703, upper bound: 3.5849247
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5849424
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848845, upper bound: 3.5849415
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848709, upper bound: 3.5848700
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848706, upper bound: 3.5848700
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5849387
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5848714
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848702, upper bound: 3.5849331
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848704, upper bound: 3.5849111
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5849401
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5848700
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848702, upper bound: 3.5849331
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848704, upper bound: 3.5848700
time: 0.48 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.25 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5848704
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5849331, upper bound: 3.5849428
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5848700
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5849401, upper bound: 3.5849145
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5848704
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5849331, upper bound: 3.5849282
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848714, upper bound: 3.5848700
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848706, upper bound: 3.5848700
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5848706
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5848709
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5849415, upper bound: 3.5848845
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5849424, upper bound: 3.5848991
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5849247, upper bound: 3.5848703
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5849322, upper bound: 3.5848702
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5849437, upper bound: 3.5848702
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5849436, upper bound: 3.5848700
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5849436
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848702, upper bound: 3.5849437
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848702, upper bound: 3.5849322
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848703, upper bound: 3.5849247
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5849424
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848845, upper bound: 3.5849415
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848709, upper bound: 3.5848700
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848706, upper bound: 3.5848700
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5849387
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5848714
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848702, upper bound: 3.5849331
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848704, upper bound: 3.5849111
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5849401
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848700, upper bound: 3.5848700
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848702, upper bound: 3.5849331
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 4, lower bound: -3.5848704, upper bound: 3.5848700

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848400
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849118, upper bound: 3.5849177
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848405
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849207, upper bound: 3.5848759
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848420
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848739, upper bound: 3.5848399
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848663, upper bound: 3.5848399
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849115, upper bound: 3.5849018
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848400
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848406, upper bound: 3.5848396
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849172, upper bound: 3.5848396
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848400
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848403
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848400, upper bound: 3.5848396
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849200, upper bound: 3.5848465
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848399, upper bound: 3.5848621
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848399, upper bound: 3.5848611
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848785, upper bound: 3.5848396
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848896, upper bound: 3.5848399
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849014, upper bound: 3.5848396
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848944, upper bound: 3.5848397
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848422, upper bound: 3.5848396
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848397
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848400, upper bound: 3.5848396
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849225
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849222
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848397, upper bound: 3.5849216
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848422
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848397, upper bound: 3.5848944
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849014
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848399, upper bound: 3.5848896
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848785
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849221
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849221
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848397, upper bound: 3.5849200
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848400
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848403, upper bound: 3.5848396
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848400, upper bound: 3.5848396
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849172
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848406
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848400, upper bound: 3.5848396
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849115
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848399, upper bound: 3.5848663
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848399, upper bound: 3.5848739
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848420, upper bound: 3.5848396
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849207
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848405, upper bound: 3.5848396
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849118
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848400, upper bound: 3.5848396
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
time: 0.44 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.34 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848400
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5849118, upper bound: 3.5849177
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848405
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5849207, upper bound: 3.5848759
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848420
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848739, upper bound: 3.5848399
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848663, upper bound: 3.5848399
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5849115, upper bound: 3.5849018
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848400
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848406, upper bound: 3.5848396
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5849172, upper bound: 3.5848396
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848400
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848403
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848400, upper bound: 3.5848396
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5849200, upper bound: 3.5848465
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848399, upper bound: 3.5848621
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848399, upper bound: 3.5848611
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848785, upper bound: 3.5848396
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848896, upper bound: 3.5848399
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5849014, upper bound: 3.5848396
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848944, upper bound: 3.5848397
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848422, upper bound: 3.5848396
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848397
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848400, upper bound: 3.5848396
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849225
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849222
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848397, upper bound: 3.5849216
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848422
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848397, upper bound: 3.5848944
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849014
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848399, upper bound: 3.5848896
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848785
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849221
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849221
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848397, upper bound: 3.5849200
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848400
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848403, upper bound: 3.5848396
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848400, upper bound: 3.5848396
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849172
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848406
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848400, upper bound: 3.5848396
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849115
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848399, upper bound: 3.5848663
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848399, upper bound: 3.5848739
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848420, upper bound: 3.5848396
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849207
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848405, upper bound: 3.5848396
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5849118
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848400, upper bound: 3.5848396
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 4, lower bound: -3.5848396, upper bound: 3.5848396

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848599, upper bound: 3.5848131
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848305
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848684, upper bound: 3.5847460
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848113
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848081
time: 0.45 seconds

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

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848625, upper bound: 3.5847460
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848555, upper bound: 3.5847460
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847654
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848687, upper bound: 3.5847460
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847914
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847892
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848428, upper bound: 3.5847460
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848630, upper bound: 3.5847460
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848132, upper bound: 3.5847460
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848683
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848678
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848132
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848630
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847461
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848346
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848428
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848292
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848182
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847892, upper bound: 3.5848132
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848684
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848687
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847654, upper bound: 3.5848138
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848555
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848625
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848081, upper bound: 3.5848571
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848011
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848130
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848684
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848131, upper bound: 3.5848599
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
time: 0.50 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.43 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5848599, upper bound: 3.5848131
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848305
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5848684, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848113
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848081
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5848625, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5848555, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847654
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5848687, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847914
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847892
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5848428, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5848630, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5848132, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848683
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848678
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848132
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848630
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847461
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848346
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848428
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848292
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848182
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847892, upper bound: 3.5848132
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848684
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848687
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847654, upper bound: 3.5848138
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848555
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5848081, upper bound: 3.5848571
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848011
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848130
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5848684
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5848131, upper bound: 3.5848599
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 4, lower bound: -3.5847460, upper bound: 3.5847460

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845566, upper bound: 3.5846315
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846566, upper bound: 3.5845538
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5846543
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5846473
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846633, upper bound: 3.5845538
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5846164
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845622, upper bound: 3.5845538
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846212, upper bound: 3.5846251
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846667, upper bound: 3.5845538
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845690
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846695, upper bound: 3.5845538
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845926
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846731, upper bound: 3.5845538
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845908
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845538, upper bound: 3.5845538
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845934, upper bound: 3.5845538
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910
1: -6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244
2: -3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980
3: -4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017
4: -2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686

Time for backsubstitution: 1.35 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.57 + 417.75 = 420.32 seconds
