## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 47.0393777385


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003)
1: (-10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643)
2: (-10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861)
3: (-15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287)
4: (-17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.66 + 1.46 = 2.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -47.1809205, upper bound: 47.1809205

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1752252, upper bound: 47.1752252
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1752252, upper bound: 47.1752252
time: 0.62 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.26 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 4, lower bound: -47.1752252, upper bound: 47.1752252
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 4, lower bound: -47.1752252, upper bound: 47.1752252

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1590057, upper bound: 47.1589816
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1590057, upper bound: 47.1589816
time: 0.38 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1680527, upper bound: 47.1685646
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1683681, upper bound: 47.1685646
time: 0.45 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.83 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.83
Output dim: 4, lower bound: -47.1590057, upper bound: 47.1589816
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.83
Output dim: 4, lower bound: -47.1590057, upper bound: 47.1589816
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.83
Output dim: 4, lower bound: -47.1680527, upper bound: 47.1685646
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.83
Output dim: 4, lower bound: -47.1683681, upper bound: 47.1685646

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1557124, upper bound: 47.1555706
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556856, upper bound: 47.1552609
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1557124, upper bound: 47.1556571
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556856, upper bound: 47.1556571
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646002, upper bound: 47.1652380
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646002, upper bound: 47.1652742
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646397, upper bound: 47.1646397
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646397, upper bound: 47.1650822
time: 0.62 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.76 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 4, lower bound: -47.1557124, upper bound: 47.1555706
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 4, lower bound: -47.1556856, upper bound: 47.1552609
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 4, lower bound: -47.1557124, upper bound: 47.1556571
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 4, lower bound: -47.1556856, upper bound: 47.1556571
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 4, lower bound: -47.1646002, upper bound: 47.1652380
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 4, lower bound: -47.1646002, upper bound: 47.1652742
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 4, lower bound: -47.1646397, upper bound: 47.1646397
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 4, lower bound: -47.1646397, upper bound: 47.1650822

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1507408, upper bound: 47.1508702
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1507408, upper bound: 47.1507408
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1550109, upper bound: 47.1550109
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1550109, upper bound: 47.1552609
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1512120, upper bound: 47.1510780
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1512197, upper bound: 47.1511529
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556856, upper bound: 47.1550109
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1550109, upper bound: 47.1556571
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1614879, upper bound: 47.1614158
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1614158, upper bound: 47.1621661
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1452852, upper bound: 47.1461312
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1452852, upper bound: 47.1461312
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1621448, upper bound: 47.1614158
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1621941, upper bound: 47.1614214
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646127, upper bound: 47.1646127
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646127, upper bound: 47.1650676
time: 0.42 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.57 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1507408, upper bound: 47.1508702
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1507408, upper bound: 47.1507408
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1550109, upper bound: 47.1550109
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1550109, upper bound: 47.1552609
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1512120, upper bound: 47.1510780
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1512197, upper bound: 47.1511529
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1556856, upper bound: 47.1550109
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1550109, upper bound: 47.1556571
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1614879, upper bound: 47.1614158
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1614158, upper bound: 47.1621661
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1452852, upper bound: 47.1461312
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1452852, upper bound: 47.1461312
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1621448, upper bound: 47.1614158
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1621941, upper bound: 47.1614214
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1646127, upper bound: 47.1646127
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.57
Output dim: 4, lower bound: -47.1646127, upper bound: 47.1650676

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1506305, upper bound: 47.1506676
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1506305, upper bound: 47.1507596
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1461477, upper bound: 47.1461477
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1461477, upper bound: 47.1461477
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0630188, upper bound: 47.0630188
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0630188, upper bound: 47.0630188
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1504877, upper bound: 47.1504891
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1504877, upper bound: 47.1507762
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1506631, upper bound: 47.1499320
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1499320, upper bound: 47.1505169
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1468756, upper bound: 47.1468756
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1468756, upper bound: 47.1476463
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1507408, upper bound: 47.1507408
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1507408, upper bound: 47.1507408
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1546084, upper bound: 47.1546084
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1546084, upper bound: 47.1552251
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1590092, upper bound: 47.1589515
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1589515, upper bound: 47.1589515
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1614106, upper bound: 47.1614106
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1614106, upper bound: 47.1621661
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -46.9435164, upper bound: 46.9435164
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -46.9435164, upper bound: 46.9435164
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1447259, upper bound: 47.1447259
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1447259, upper bound: 47.1456433
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -46.9426080, upper bound: 46.9426080
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -46.9426080, upper bound: 46.9426080
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1614134, upper bound: 47.1614151
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1614134, upper bound: 47.1614106
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646035, upper bound: 47.1646035
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646035, upper bound: 47.1646035
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1418720, upper bound: 47.1427807
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1418720, upper bound: 47.1427807
time: 0.41 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1506305, upper bound: 47.1506676
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1506305, upper bound: 47.1507596
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1461477, upper bound: 47.1461477
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1461477, upper bound: 47.1461477
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.0630188, upper bound: 47.0630188
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.0630188, upper bound: 47.0630188
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1504877, upper bound: 47.1504891
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1504877, upper bound: 47.1507762
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1506631, upper bound: 47.1499320
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1499320, upper bound: 47.1505169
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1468756, upper bound: 47.1468756
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1468756, upper bound: 47.1476463
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1507408, upper bound: 47.1507408
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1507408, upper bound: 47.1507408
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1546084, upper bound: 47.1546084
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1546084, upper bound: 47.1552251
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1590092, upper bound: 47.1589515
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1589515, upper bound: 47.1589515
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1614106, upper bound: 47.1614106
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1614106, upper bound: 47.1621661
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 4, lower bound: -46.9435164, upper bound: 46.9435164
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 4, lower bound: -46.9435164, upper bound: 46.9435164
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1447259, upper bound: 47.1447259
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1447259, upper bound: 47.1456433
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 4, lower bound: -46.9426080, upper bound: 46.9426080
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 4, lower bound: -46.9426080, upper bound: 46.9426080
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1614134, upper bound: 47.1614151
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1614134, upper bound: 47.1614106
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1646035, upper bound: 47.1646035
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1646035, upper bound: 47.1646035
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1418720, upper bound: 47.1427807
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 4, lower bound: -47.1418720, upper bound: 47.1427807

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1461909, upper bound: 47.1461909
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1461909, upper bound: 47.1462509
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1506305, upper bound: 47.1506305
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1506305, upper bound: 47.1507596
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1416888, upper bound: 47.1416888
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1416888, upper bound: 47.1416888
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1267361, upper bound: 47.1267361
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1267361, upper bound: 47.1267361
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0628694, upper bound: 47.0628694
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0628694, upper bound: 47.0628694
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0498885, upper bound: 47.0498885
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0498885, upper bound: 47.0498885
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1499320, upper bound: 47.1499320
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1499320, upper bound: 47.1499320
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1468756, upper bound: 47.1468756
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1468756, upper bound: 47.1472873
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1267361, upper bound: 47.1267361
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1267361, upper bound: 47.1267361
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1267361, upper bound: 47.1267361
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1267361, upper bound: 47.1267361
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1416916, upper bound: 47.1416888
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1416916, upper bound: 47.1416888
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1416888, upper bound: 47.1418518
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1416888, upper bound: 47.1416888
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1419959, upper bound: 47.1419959
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1419959, upper bound: 47.1419959
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1545187, upper bound: 47.1545187
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1545187, upper bound: 47.1545187
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1520654, upper bound: 47.1526830
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1520654, upper bound: 47.1526377
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1369359, upper bound: 47.1369359
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1369359, upper bound: 47.1369359
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1590048, upper bound: 47.1589515
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1589515, upper bound: 47.1589515
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1613560, upper bound: 47.1613560
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1613560, upper bound: 47.1613560
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1584257, upper bound: 47.1585815
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1584257, upper bound: 47.1584257
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1446471, upper bound: 47.1446471
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1446471, upper bound: 47.1446471
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1447259, upper bound: 47.1447259
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1447259, upper bound: 47.1456433
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1614106, upper bound: 47.1614121
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1614106, upper bound: 47.1614128
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1584257, upper bound: 47.1584257
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1584257, upper bound: 47.1584257
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1604517, upper bound: 47.1604517
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1604517, upper bound: 47.1604517
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1604517, upper bound: 47.1604517
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1604517, upper bound: 47.1604517
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1379637, upper bound: 47.1382225
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1379637, upper bound: 47.1379637
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1369359, upper bound: 47.1379222
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1369359, upper bound: 47.1379610
time: 0.74 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.14 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1461909, upper bound: 47.1461909
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1461909, upper bound: 47.1462509
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1506305, upper bound: 47.1506305
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1506305, upper bound: 47.1507596
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1416888, upper bound: 47.1416888
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1416888, upper bound: 47.1416888
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1267361, upper bound: 47.1267361
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1267361, upper bound: 47.1267361
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.0628694, upper bound: 47.0628694
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.0628694, upper bound: 47.0628694
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.0498885, upper bound: 47.0498885
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.0498885, upper bound: 47.0498885
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1499320, upper bound: 47.1499320
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1499320, upper bound: 47.1499320
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1468756, upper bound: 47.1468756
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1468756, upper bound: 47.1472873
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1267361, upper bound: 47.1267361
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1267361, upper bound: 47.1267361
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1267361, upper bound: 47.1267361
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1267361, upper bound: 47.1267361
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1416916, upper bound: 47.1416888
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1416916, upper bound: 47.1416888
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1416888, upper bound: 47.1418518
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1416888, upper bound: 47.1416888
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1419959, upper bound: 47.1419959
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1419959, upper bound: 47.1419959
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1545187, upper bound: 47.1545187
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1545187, upper bound: 47.1545187
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1520654, upper bound: 47.1526830
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1520654, upper bound: 47.1526377
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1369359, upper bound: 47.1369359
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1369359, upper bound: 47.1369359
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1590048, upper bound: 47.1589515
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1589515, upper bound: 47.1589515
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1613560, upper bound: 47.1613560
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1613560, upper bound: 47.1613560
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1584257, upper bound: 47.1585815
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1584257, upper bound: 47.1584257
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1446471, upper bound: 47.1446471
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1446471, upper bound: 47.1446471
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1447259, upper bound: 47.1447259
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1447259, upper bound: 47.1456433
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1614106, upper bound: 47.1614121
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1614106, upper bound: 47.1614128
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1584257, upper bound: 47.1584257
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1584257, upper bound: 47.1584257
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1604517, upper bound: 47.1604517
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1604517, upper bound: 47.1604517
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1604517, upper bound: 47.1604517
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1604517, upper bound: 47.1604517
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1379637, upper bound: 47.1382225
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1379637, upper bound: 47.1379637
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1369359, upper bound: 47.1379222
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 4, lower bound: -47.1369359, upper bound: 47.1379610

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1415825, upper bound: 47.1415795
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1416268
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1314219, upper bound: 47.1314219
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1314219, upper bound: 47.1314219
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1461909, upper bound: 47.1461909
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1461909, upper bound: 47.1463386
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1416888, upper bound: 47.1416888
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1416888, upper bound: 47.1416888
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0615659, upper bound: 47.0615659
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0615659, upper bound: 47.0615659
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0615659, upper bound: 47.0615659
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0615659, upper bound: 47.0615659
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0485572, upper bound: 47.0485572
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0485572, upper bound: 47.0485572
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -46.9417455, upper bound: 46.9417455
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -46.9417455, upper bound: 46.9417455
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1463121, upper bound: 47.1463121
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1463121, upper bound: 47.1463121
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1467836, upper bound: 47.1467836
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1467836, upper bound: 47.1467836
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1467836, upper bound: 47.1471924
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1467836, upper bound: 47.1470640
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266755, upper bound: 47.1266755
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266755, upper bound: 47.1266755
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266755, upper bound: 47.1266755
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266755, upper bound: 47.1266755
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1394982, upper bound: 47.1394982
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1394982, upper bound: 47.1394982
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1394982, upper bound: 47.1394982
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1396946, upper bound: 47.1394982
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1394982, upper bound: 47.1397045
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1394982, upper bound: 47.1396813
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1394982, upper bound: 47.1394982
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1394982, upper bound: 47.1394982
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1379637, upper bound: 47.1379637
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1379637, upper bound: 47.1379637
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1379637, upper bound: 47.1379637
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1379637, upper bound: 47.1379637
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1511393, upper bound: 47.1511393
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1511393, upper bound: 47.1511393
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1498410, upper bound: 47.1498410
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1498410, upper bound: 47.1498410
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1252190, upper bound: 47.1252190
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1252190, upper bound: 47.1252190
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1519749, upper bound: 47.1525442
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1519749, upper bound: 47.1525466
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1325068, upper bound: 47.1325068
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1325709, upper bound: 47.1325068
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1368594, upper bound: 47.1368594
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1368594, upper bound: 47.1368594
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1513056, upper bound: 47.1513056
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1513056, upper bound: 47.1513056
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1558792, upper bound: 47.1558792
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1558792, upper bound: 47.1558792
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1583603, upper bound: 47.1583603
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1583603, upper bound: 47.1583603
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1583603, upper bound: 47.1583603
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1583603, upper bound: 47.1583603
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1556655
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1558971
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1556655
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1556655
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1402887, upper bound: 47.1402887
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1402887, upper bound: 47.1402887
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1395596, upper bound: 47.1395596
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1395596, upper bound: 47.1395596
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1396405, upper bound: 47.1396405
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1396405, upper bound: 47.1396405
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1613560, upper bound: 47.1613576
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1621317, upper bound: 47.1613560
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1587907, upper bound: 47.1587923
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1587907, upper bound: 47.1587907
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1583603, upper bound: 47.1583603
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1583603, upper bound: 47.1583603
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1277031, upper bound: 47.1277031
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1277031, upper bound: 47.1277031
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1584369, upper bound: 47.1584369
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1584369, upper bound: 47.1584369
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1578492, upper bound: 47.1578492
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1578492, upper bound: 47.1578492
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1362458, upper bound: 47.1362458
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1362458, upper bound: 47.1362458
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1578492, upper bound: 47.1578492
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1578492, upper bound: 47.1578492
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1338684
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1329752
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1339582
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1324285
time: 0.50 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.30 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1415825, upper bound: 47.1415795
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1416268
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1314219, upper bound: 47.1314219
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1314219, upper bound: 47.1314219
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1461909, upper bound: 47.1461909
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1461909, upper bound: 47.1463386
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1416888, upper bound: 47.1416888
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1416888, upper bound: 47.1416888
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.0615659, upper bound: 47.0615659
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.0615659, upper bound: 47.0615659
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.0615659, upper bound: 47.0615659
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.0615659, upper bound: 47.0615659
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.0485572, upper bound: 47.0485572
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.0485572, upper bound: 47.0485572
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.30
Output dim: 4, lower bound: -46.9417455, upper bound: 46.9417455
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.30
Output dim: 4, lower bound: -46.9417455, upper bound: 46.9417455
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1463121, upper bound: 47.1463121
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1463121, upper bound: 47.1463121
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1467836, upper bound: 47.1467836
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1467836, upper bound: 47.1467836
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1467836, upper bound: 47.1471924
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1467836, upper bound: 47.1470640
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1266755, upper bound: 47.1266755
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1266755, upper bound: 47.1266755
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1266755, upper bound: 47.1266755
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1266755, upper bound: 47.1266755
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1394982, upper bound: 47.1394982
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1394982, upper bound: 47.1394982
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1394982, upper bound: 47.1394982
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1396946, upper bound: 47.1394982
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1394982, upper bound: 47.1397045
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1394982, upper bound: 47.1396813
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1394982, upper bound: 47.1394982
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1394982, upper bound: 47.1394982
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1379637, upper bound: 47.1379637
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1379637, upper bound: 47.1379637
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1379637, upper bound: 47.1379637
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1379637, upper bound: 47.1379637
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1511393, upper bound: 47.1511393
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1511393, upper bound: 47.1511393
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1498410, upper bound: 47.1498410
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1498410, upper bound: 47.1498410
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1252190, upper bound: 47.1252190
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1252190, upper bound: 47.1252190
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1519749, upper bound: 47.1525442
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1519749, upper bound: 47.1525466
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1325068, upper bound: 47.1325068
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1325709, upper bound: 47.1325068
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1368594, upper bound: 47.1368594
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1368594, upper bound: 47.1368594
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1513056, upper bound: 47.1513056
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1513056, upper bound: 47.1513056
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1558792, upper bound: 47.1558792
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1558792, upper bound: 47.1558792
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1583603, upper bound: 47.1583603
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1583603, upper bound: 47.1583603
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1583603, upper bound: 47.1583603
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1583603, upper bound: 47.1583603
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1556655
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1558971
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1556655
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1556655
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1402887, upper bound: 47.1402887
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1402887, upper bound: 47.1402887
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1395596, upper bound: 47.1395596
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1395596, upper bound: 47.1395596
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1396405, upper bound: 47.1396405
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1396405, upper bound: 47.1396405
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1314812, upper bound: 47.1314812
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1613560, upper bound: 47.1613576
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1621317, upper bound: 47.1613560
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1587907, upper bound: 47.1587923
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1587907, upper bound: 47.1587907
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1583603, upper bound: 47.1583603
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1583603, upper bound: 47.1583603
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1277031, upper bound: 47.1277031
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1277031, upper bound: 47.1277031
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1584369, upper bound: 47.1584369
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1584369, upper bound: 47.1584369
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1578492, upper bound: 47.1578492
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1578492, upper bound: 47.1578492
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1362458, upper bound: 47.1362458
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1362458, upper bound: 47.1362458
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1578492, upper bound: 47.1578492
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1578492, upper bound: 47.1578492
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1338684
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1329752
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1339582
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.30
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1324285

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1332994, upper bound: 47.1332994
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1332994, upper bound: 47.1332994
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1416268
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266048, upper bound: 47.1266048
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266048, upper bound: 47.1266048
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1314219, upper bound: 47.1314219
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1314219, upper bound: 47.1314219
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1434356, upper bound: 47.1436074
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1434356, upper bound: 47.1434356
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1394964, upper bound: 47.1394964
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1394964, upper bound: 47.1394964
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1333423, upper bound: 47.1333423
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1333423, upper bound: 47.1333423
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266048, upper bound: 47.1266048
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266048, upper bound: 47.1266048
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1203412, upper bound: 47.1203412
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1203412, upper bound: 47.1203412
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -46.9389868, upper bound: 46.9389868
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -46.9389868, upper bound: 46.9389868
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0484093, upper bound: 47.0484093
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0484093, upper bound: 47.0484093
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -46.9389868, upper bound: 46.9389868
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -46.9389868, upper bound: 46.9389868
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -46.9389868, upper bound: 46.9389868
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -46.9389868, upper bound: 46.9389868
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0484093, upper bound: 47.0484093
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0484093, upper bound: 47.0484093
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0484093, upper bound: 47.0484093
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0484093, upper bound: 47.0484093
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1362378, upper bound: 47.1362378
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1362378, upper bound: 47.1362378
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228179, upper bound: 47.1228179
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228179, upper bound: 47.1228179
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228179, upper bound: 47.1228179
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228179, upper bound: 47.1228179
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1203412, upper bound: 47.1203412
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1203412, upper bound: 47.1203412
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1462210, upper bound: 47.1462210
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1462210, upper bound: 47.1462210
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1440064, upper bound: 47.1443991
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1440064, upper bound: 47.1440064
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1440064, upper bound: 47.1442844
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1440064, upper bound: 47.1440064
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1203600, upper bound: 47.1203600
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1203600, upper bound: 47.1203600
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228179, upper bound: 47.1228179
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228179, upper bound: 47.1228179
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266048, upper bound: 47.1266048
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266048, upper bound: 47.1266048
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1203600, upper bound: 47.1203600
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1203600, upper bound: 47.1203600
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1393958, upper bound: 47.1393958
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1393958, upper bound: 47.1393958
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1393958, upper bound: 47.1393958
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1393958, upper bound: 47.1393958
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1395925, upper bound: 47.1393958
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1393958, upper bound: 47.1393958
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1394964, upper bound: 47.1394964
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1394964, upper bound: 47.1397045
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1393958, upper bound: 47.1393958
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1393958, upper bound: 47.1393958
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1290747, upper bound: 47.1290747
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1290747, upper bound: 47.1290747
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1314219, upper bound: 47.1314219
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1314219, upper bound: 47.1314219
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1252190, upper bound: 47.1252190
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1252190, upper bound: 47.1252190
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1252190, upper bound: 47.1252190
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1252190, upper bound: 47.1252190
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1332301, upper bound: 47.1332301
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1332301, upper bound: 47.1332301
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1378812, upper bound: 47.1378812
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1378812, upper bound: 47.1378812
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1333423, upper bound: 47.1333423
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1333423, upper bound: 47.1333423
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1332301, upper bound: 47.1332301
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1332301, upper bound: 47.1332301
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1483096, upper bound: 47.1483096
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1483096, upper bound: 47.1483096
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1483096, upper bound: 47.1483096
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1483096, upper bound: 47.1483096
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1470742, upper bound: 47.1470742
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1470742, upper bound: 47.1470742
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1395596, upper bound: 47.1395596
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1395596, upper bound: 47.1395596
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1251577, upper bound: 47.1251577
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1251577, upper bound: 47.1251577
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1203412, upper bound: 47.1203412
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1203412, upper bound: 47.1203412
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1483096, upper bound: 47.1483096
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1483096, upper bound: 47.1488968
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1402158, upper bound: 47.1402158
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1402158, upper bound: 47.1402158
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1316265, upper bound: 47.1316265
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1316265, upper bound: 47.1316265
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1290747, upper bound: 47.1290747
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1290747, upper bound: 47.1290747
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1361626, upper bound: 47.1361626
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1361626, upper bound: 47.1361626
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1324380, upper bound: 47.1324380
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1324380, upper bound: 47.1324380
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1512413, upper bound: 47.1512413
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1513134, upper bound: 47.1512413
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1324285
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1324285
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1333423, upper bound: 47.1333423
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1333423, upper bound: 47.1333423
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1558111, upper bound: 47.1558111
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1558111, upper bound: 47.1558111
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556015, upper bound: 47.1556015
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556015, upper bound: 47.1556015
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1276438, upper bound: 47.1276438
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1276438, upper bound: 47.1276438
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1510529, upper bound: 47.1510529
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1510529, upper bound: 47.1510529
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1583375, upper bound: 47.1583375
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1583375, upper bound: 47.1583375
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556015, upper bound: 47.1556015
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556015, upper bound: 47.1556015
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1556655
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1558971
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556015, upper bound: 47.1556015
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556015, upper bound: 47.1556015
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1556655
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1556655
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1362323, upper bound: 47.1362323
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1362323, upper bound: 47.1362323
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1357171, upper bound: 47.1357171
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1357171, upper bound: 47.1357171
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1395596, upper bound: 47.1395596
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1395596, upper bound: 47.1395596
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266751, upper bound: 47.1266751
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266751, upper bound: 47.1266751
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1357128, upper bound: 47.1357128
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1357128, upper bound: 47.1357128
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1412210, upper bound: 47.1412210
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1412210, upper bound: 47.1412210
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1412210, upper bound: 47.1412210
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1412210, upper bound: 47.1412210
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1587359, upper bound: 47.1587373
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1587359, upper bound: 47.1587359
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1587359, upper bound: 47.1587359
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1587359, upper bound: 47.1587359
time: 0.74 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.58 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1332994, upper bound: 47.1332994
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1332994, upper bound: 47.1332994
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1416268
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266048, upper bound: 47.1266048
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266048, upper bound: 47.1266048
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1314219, upper bound: 47.1314219
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1314219, upper bound: 47.1314219
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1434356, upper bound: 47.1436074
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1434356, upper bound: 47.1434356
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1394964, upper bound: 47.1394964
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1394964, upper bound: 47.1394964
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1333423, upper bound: 47.1333423
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1333423, upper bound: 47.1333423
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266048, upper bound: 47.1266048
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266048, upper bound: 47.1266048
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1203412, upper bound: 47.1203412
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1203412, upper bound: 47.1203412
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 4, lower bound: -46.9389868, upper bound: 46.9389868
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 4, lower bound: -46.9389868, upper bound: 46.9389868
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.0484093, upper bound: 47.0484093
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.0484093, upper bound: 47.0484093
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 4, lower bound: -46.9389868, upper bound: 46.9389868
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 4, lower bound: -46.9389868, upper bound: 46.9389868
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 4, lower bound: -46.9389868, upper bound: 46.9389868
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 4, lower bound: -46.9389868, upper bound: 46.9389868
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.0484093, upper bound: 47.0484093
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.0484093, upper bound: 47.0484093
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.0484093, upper bound: 47.0484093
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.0484093, upper bound: 47.0484093
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1362378, upper bound: 47.1362378
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1362378, upper bound: 47.1362378
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228179, upper bound: 47.1228179
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228179, upper bound: 47.1228179
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228179, upper bound: 47.1228179
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228179, upper bound: 47.1228179
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1203412, upper bound: 47.1203412
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1203412, upper bound: 47.1203412
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1415795, upper bound: 47.1415795
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1462210, upper bound: 47.1462210
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1462210, upper bound: 47.1462210
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1440064, upper bound: 47.1443991
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1440064, upper bound: 47.1440064
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1440064, upper bound: 47.1442844
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1440064, upper bound: 47.1440064
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1203600, upper bound: 47.1203600
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1203600, upper bound: 47.1203600
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228179, upper bound: 47.1228179
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228179, upper bound: 47.1228179
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228843, upper bound: 47.1228843
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266048, upper bound: 47.1266048
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266048, upper bound: 47.1266048
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1203600, upper bound: 47.1203600
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1203600, upper bound: 47.1203600
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1228243, upper bound: 47.1228243
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1393958, upper bound: 47.1393958
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1393958, upper bound: 47.1393958
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1393958, upper bound: 47.1393958
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1393958, upper bound: 47.1393958
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1395925, upper bound: 47.1393958
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1393958, upper bound: 47.1393958
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1394964, upper bound: 47.1394964
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1394964, upper bound: 47.1397045
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1165867, upper bound: 47.1165867
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1393958, upper bound: 47.1393958
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1393958, upper bound: 47.1393958
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1290747, upper bound: 47.1290747
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1290747, upper bound: 47.1290747
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1314219, upper bound: 47.1314219
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1314219, upper bound: 47.1314219
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1252190, upper bound: 47.1252190
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1252190, upper bound: 47.1252190
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1252190, upper bound: 47.1252190
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1252190, upper bound: 47.1252190
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1332301, upper bound: 47.1332301
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1332301, upper bound: 47.1332301
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1378812, upper bound: 47.1378812
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1378812, upper bound: 47.1378812
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1333423, upper bound: 47.1333423
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1333423, upper bound: 47.1333423
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1332301, upper bound: 47.1332301
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1332301, upper bound: 47.1332301
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1483096, upper bound: 47.1483096
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1483096, upper bound: 47.1483096
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1483096, upper bound: 47.1483096
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1483096, upper bound: 47.1483096
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1470742, upper bound: 47.1470742
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1470742, upper bound: 47.1470742
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1395596, upper bound: 47.1395596
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1395596, upper bound: 47.1395596
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1251577, upper bound: 47.1251577
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1251577, upper bound: 47.1251577
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1203412, upper bound: 47.1203412
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1203412, upper bound: 47.1203412
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1483096, upper bound: 47.1483096
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1483096, upper bound: 47.1488968
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1402158, upper bound: 47.1402158
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1402158, upper bound: 47.1402158
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1316265, upper bound: 47.1316265
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1316265, upper bound: 47.1316265
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1290747, upper bound: 47.1290747
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1290747, upper bound: 47.1290747
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1361626, upper bound: 47.1361626
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1361626, upper bound: 47.1361626
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1324380, upper bound: 47.1324380
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1324380, upper bound: 47.1324380
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1512413, upper bound: 47.1512413
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1513134, upper bound: 47.1512413
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1324285
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1324285
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1333423, upper bound: 47.1333423
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1333423, upper bound: 47.1333423
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1558111, upper bound: 47.1558111
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1558111, upper bound: 47.1558111
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1556015, upper bound: 47.1556015
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1556015, upper bound: 47.1556015
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1276438, upper bound: 47.1276438
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1276438, upper bound: 47.1276438
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1510529, upper bound: 47.1510529
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1510529, upper bound: 47.1510529
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1583375, upper bound: 47.1583375
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1583375, upper bound: 47.1583375
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1556015, upper bound: 47.1556015
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1556015, upper bound: 47.1556015
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1556655
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1558971
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1556015, upper bound: 47.1556015
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1556015, upper bound: 47.1556015
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1556655
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1556655, upper bound: 47.1556655
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1362323, upper bound: 47.1362323
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1362323, upper bound: 47.1362323
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1357171, upper bound: 47.1357171
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1357171, upper bound: 47.1357171
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1395596, upper bound: 47.1395596
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1395596, upper bound: 47.1395596
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266751, upper bound: 47.1266751
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266751, upper bound: 47.1266751
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1357128, upper bound: 47.1357128
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1357128, upper bound: 47.1357128
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1266652, upper bound: 47.1266652
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1412210, upper bound: 47.1412210
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1412210, upper bound: 47.1412210
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1412210, upper bound: 47.1412210
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1412210, upper bound: 47.1412210
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1587359, upper bound: 47.1587373
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1587359, upper bound: 47.1587359
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1587359, upper bound: 47.1587359
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 4, lower bound: -47.1587359, upper bound: 47.1587359
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1583603, upper bound: 47.1583603
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1583603, upper bound: 47.1583603
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1277031, upper bound: 47.1277031
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1277031, upper bound: 47.1277031
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1584369, upper bound: 47.1584369
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1584369, upper bound: 47.1584369
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1578492, upper bound: 47.1578492
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1578492, upper bound: 47.1578492
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1362458, upper bound: 47.1362458
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1362458, upper bound: 47.1362458
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1578492, upper bound: 47.1578492
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1578492, upper bound: 47.1578492
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1276249, upper bound: 47.1276249
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1338684
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1329752
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1339582
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -47.1324285, upper bound: 47.1324285

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.11 + 418.83 = 420.94 seconds
