## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0015192279999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823)
1: (-0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0038504, 0.0038504)
2: (0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026485, 0.0026485)
3: (-0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0046414, 0.0046414)
4: (-0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0040695, 0.0040695)
5: (0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447)
6: (-0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0056837, 0.0056837)
7: (0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0039407, 0.0039407)
8: (-0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088)
9: (0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 2.76 = 4.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0016162, upper bound: 0.0016162

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015515, upper bound: 0.0015544
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015547, upper bound: 0.0015517
time: 1.69 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.69 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.69
Output dim: 2, lower bound: -0.0015515, upper bound: 0.0015544
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.69
Output dim: 2, lower bound: -0.0015547, upper bound: 0.0015517

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036995, 0.0037220
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025793, 0.0025932
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044873, 0.0044613
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039259, 0.0039487
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0055069, 0.0054739
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0038087, 0.0037856
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015506, upper bound: 0.0015530
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015499, upper bound: 0.0015536
time: 1.77 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037220, 0.0036995
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025932, 0.0025793
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044613, 0.0044873
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039487, 0.0039259
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054739, 0.0055069
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037856, 0.0038087
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015499, upper bound: 0.0015499
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015499, upper bound: 0.0015506
time: 1.90 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.11 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 2, lower bound: -0.0015506, upper bound: 0.0015530
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 2, lower bound: -0.0015499, upper bound: 0.0015536
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 2, lower bound: -0.0015499, upper bound: 0.0015499
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.11
Output dim: 2, lower bound: -0.0015499, upper bound: 0.0015506

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036968, 0.0037193
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025776, 0.0025915
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044841, 0.0044581
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039231, 0.0039460
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0055030, 0.0054700
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0038059, 0.0037828
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015450, upper bound: 0.0015502
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015476, upper bound: 0.0015452
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036967, 0.0037193
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025775, 0.0025915
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044841, 0.0044580
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039230, 0.0039459
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0055029, 0.0054698
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0038059, 0.0037827
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015438, upper bound: 0.0015508
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015469, upper bound: 0.0015466
time: 1.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037193, 0.0036967
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025915, 0.0025775
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044580, 0.0044841
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039459, 0.0039230
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054698, 0.0055029
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037827, 0.0038059
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015466, upper bound: 0.0015469
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015508, upper bound: 0.0015438
time: 2.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037193, 0.0036968
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025915, 0.0025776
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044581, 0.0044841
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039460, 0.0039231
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054700, 0.0055030
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037828, 0.0038059
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015452, upper bound: 0.0015476
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015502, upper bound: 0.0015451
time: 1.94 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.05 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 2, lower bound: -0.0015450, upper bound: 0.0015502
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 2, lower bound: -0.0015476, upper bound: 0.0015452
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 2, lower bound: -0.0015438, upper bound: 0.0015508
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 2, lower bound: -0.0015469, upper bound: 0.0015466
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 2, lower bound: -0.0015466, upper bound: 0.0015469
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 2, lower bound: -0.0015508, upper bound: 0.0015438
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 2, lower bound: -0.0015452, upper bound: 0.0015476
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.05
Output dim: 2, lower bound: -0.0015502, upper bound: 0.0015451

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036843, 0.0037105
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025712, 0.0025875
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044744, 0.0044441
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039122, 0.0039389
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054922, 0.0054537
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037970, 0.0037700
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015351, upper bound: 0.0015486
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015350, upper bound: 0.0015406
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036880, 0.0037068
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025735, 0.0025852
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044701, 0.0044484
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039160, 0.0039350
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054867, 0.0054591
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037931, 0.0037738
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015369, upper bound: 0.0015435
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015460, upper bound: 0.0015367
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036842, 0.0037105
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025712, 0.0025875
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044744, 0.0044440
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039121, 0.0039388
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054922, 0.0054535
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037969, 0.0037699
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015350, upper bound: 0.0015492
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015350, upper bound: 0.0015408
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036879, 0.0037067
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025735, 0.0025851
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044701, 0.0044482
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039159, 0.0039350
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054867, 0.0054589
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037931, 0.0037737
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015365, upper bound: 0.0015449
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015366, upper bound: 0.0015370
time: 1.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037067, 0.0036879
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025851, 0.0025735
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044482, 0.0044701
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039350, 0.0039159
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054589, 0.0054867
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037737, 0.0037931
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015370, upper bound: 0.0015453
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015350, upper bound: 0.0015365
time: 1.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037105, 0.0036842
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025875, 0.0025712
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044440, 0.0044744
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039388, 0.0039121
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054535, 0.0054922
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037699, 0.0037969
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015366, upper bound: 0.0015422
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015492, upper bound: 0.0015350
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037068, 0.0036880
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025852, 0.0025735
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044484, 0.0044701
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039350, 0.0039160
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054591, 0.0054867
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037738, 0.0037931
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015350, upper bound: 0.0015460
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015350, upper bound: 0.0015368
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037105, 0.0036843
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025875, 0.0025712
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044441, 0.0044744
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039389, 0.0039122
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054537, 0.0054922
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037700, 0.0037970
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015407, upper bound: 0.0015434
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015486, upper bound: 0.0015352
time: 1.73 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.87 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015351, upper bound: 0.0015486
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015350, upper bound: 0.0015406
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015369, upper bound: 0.0015435
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015460, upper bound: 0.0015367
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015350, upper bound: 0.0015492
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015350, upper bound: 0.0015408
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015365, upper bound: 0.0015449
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015366, upper bound: 0.0015370
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015370, upper bound: 0.0015453
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015350, upper bound: 0.0015365
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015366, upper bound: 0.0015422
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015492, upper bound: 0.0015350
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015350, upper bound: 0.0015460
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015350, upper bound: 0.0015368
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015407, upper bound: 0.0015434
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0015486, upper bound: 0.0015352

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036540, 0.0036888
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025614, 0.0025830
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044572, 0.0044169
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038695, 0.0039049
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054442, 0.0053931
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037740, 0.0037383
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0015135
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0014622
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036639, 0.0036802
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025676, 0.0025777
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044473, 0.0044283
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038796, 0.0038962
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054316, 0.0054076
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037652, 0.0037484
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0015049
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0014583
time: 2.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036577, 0.0036850
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025637, 0.0025806
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044528, 0.0044212
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038733, 0.0039010
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054386, 0.0053985
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037701, 0.0037421
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014555, upper bound: 0.0015091
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014555, upper bound: 0.0014590
time: 1.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036676, 0.0036765
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025698, 0.0025754
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044429, 0.0044326
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038833, 0.0038924
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054261, 0.0054130
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037614, 0.0037522
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014555, upper bound: 0.0015013
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015106, upper bound: 0.0014545
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036539, 0.0036889
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025614, 0.0025831
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044573, 0.0044168
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038694, 0.0039050
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054443, 0.0053929
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037741, 0.0037382
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0015139
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0014632
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036637, 0.0036802
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025674, 0.0025777
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044472, 0.0044281
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038794, 0.0038961
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054316, 0.0054073
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037652, 0.0037483
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0015049
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0014583
time: 2.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036576, 0.0036850
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025637, 0.0025807
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044529, 0.0044211
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038732, 0.0039011
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054387, 0.0053984
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037702, 0.0037420
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015100
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0014601
time: 1.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036674, 0.0036765
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025697, 0.0025753
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044429, 0.0044324
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038831, 0.0038923
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054261, 0.0054128
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037614, 0.0037520
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015014
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015104, upper bound: 0.0014546
time: 1.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036765, 0.0036674
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025753, 0.0025697
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044324, 0.0044429
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038923, 0.0038831
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054128, 0.0054261
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037520, 0.0037614
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0015104
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015013, upper bound: 0.0014601
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036851, 0.0036576
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025807, 0.0025637
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044211, 0.0044529
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039011, 0.0038732
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0053984, 0.0054387
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037420, 0.0037702
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014601, upper bound: 0.0015009
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0014554
time: 1.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036802, 0.0036637
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025777, 0.0025674
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044281, 0.0044472
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038961, 0.0038794
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054073, 0.0054316
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037483, 0.0037652
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014555, upper bound: 0.0015078
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014555, upper bound: 0.0014579
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036889, 0.0036539
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025830, 0.0025614
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044168, 0.0044573
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039050, 0.0038694
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0053929, 0.0054443
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037382, 0.0037741
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014555, upper bound: 0.0014994
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015138, upper bound: 0.0014534
time: 2.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036765, 0.0036676
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025754, 0.0025698
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044326, 0.0044429
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038924, 0.0038833
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054130, 0.0054261
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037522, 0.0037614
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0015106
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0014610
time: 1.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036850, 0.0036577
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025806, 0.0025637
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044212, 0.0044528
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039010, 0.0038733
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0053985, 0.0054386
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037421, 0.0037701
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0015011
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0014555
time: 2.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036802, 0.0036639
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025777, 0.0025676
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044283, 0.0044473
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038962, 0.0038796
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054076, 0.0054316
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037484, 0.0037652
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015083
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015049, upper bound: 0.0014589
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036888, 0.0036540
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025830, 0.0025614
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044169, 0.0044572
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039049, 0.0038695
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0053931, 0.0054442
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037383, 0.0037740
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0014995
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015135, upper bound: 0.0014535
time: 1.80 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.16 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0015135
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0014622
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0015049
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0014583
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014555, upper bound: 0.0015091
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014555, upper bound: 0.0014590
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014555, upper bound: 0.0015013
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0015106, upper bound: 0.0014545
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0015139
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0014632
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0015049
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0014583
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015100
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0014601
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015014
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0015104, upper bound: 0.0014546
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0015104
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0015013, upper bound: 0.0014601
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014601, upper bound: 0.0015009
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0014554
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014555, upper bound: 0.0015078
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014555, upper bound: 0.0014579
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014555, upper bound: 0.0014994
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0015138, upper bound: 0.0014534
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0015106
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0014610
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0015011
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014534, upper bound: 0.0014555
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015083
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0015049, upper bound: 0.0014589
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0014995
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.16
Output dim: 2, lower bound: -0.0015135, upper bound: 0.0014535

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.26 + 160.08 = 164.34 seconds
