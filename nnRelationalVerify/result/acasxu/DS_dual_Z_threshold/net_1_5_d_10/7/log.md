## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 339.77104719722996


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423)
1: (-124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621)
2: (-105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148)
3: (-110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960)
4: (-94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.70 + 2.24 = 2.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -339.8050277, upper bound: 339.8050277

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8040862, upper bound: 339.8040862
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8040862, upper bound: 339.8040862
time: 1.13 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.33 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.33
Output dim: 0, lower bound: -339.8040862, upper bound: 339.8040862
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.33
Output dim: 0, lower bound: -339.8040862, upper bound: 339.8040862

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7946390, upper bound: 339.7946390
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7946390, upper bound: 339.7946390
time: 1.08 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7946390, upper bound: 339.7946390
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7946390, upper bound: 339.7946390
time: 1.08 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.88 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -339.7946390, upper bound: 339.7946390
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -339.7946390, upper bound: 339.7946390
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -339.7946390, upper bound: 339.7946390
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -339.7946390, upper bound: 339.7946390

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635876, upper bound: 339.7635739
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635876, upper bound: 339.7635739
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635878, upper bound: 339.7635851
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635739, upper bound: 339.7635851
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635851, upper bound: 339.7635878
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635851, upper bound: 339.7635878
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635739, upper bound: 339.7635876
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635739, upper bound: 339.7635876
time: 0.88 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.48 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -339.7635876, upper bound: 339.7635739
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -339.7635876, upper bound: 339.7635739
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -339.7635878, upper bound: 339.7635851
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -339.7635739, upper bound: 339.7635851
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -339.7635851, upper bound: 339.7635878
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -339.7635851, upper bound: 339.7635878
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -339.7635739, upper bound: 339.7635876
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -339.7635739, upper bound: 339.7635876

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.95 + 18.54 = 21.48 seconds
