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
execution time: IAR + RelationalAnalysis = 0.99 + 2.64 = 3.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0016162, upper bound: 0.0016162

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0016008, upper bound: 0.0016039
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0016008, upper bound: 0.0016008
time: 1.89 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.70 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.70
Output dim: 2, lower bound: -0.0016008, upper bound: 0.0016039
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.70
Output dim: 2, lower bound: -0.0016008, upper bound: 0.0016008

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0038258, 0.0038289
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026485, 0.0026485
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0046153, 0.0046117
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0040499, 0.0040531
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0056596, 0.0056551
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0039186, 0.0039154
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015587, upper bound: 0.0015610
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015587, upper bound: 0.0015611
time: 1.83 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0038289, 0.0038258
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026485, 0.0026485
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0046117, 0.0046153
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0040531, 0.0040499
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0056551, 0.0056596
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0039154, 0.0039186
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015942, upper bound: 0.0015992
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0016023, upper bound: 0.0015899
time: 1.78 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.52 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 2, lower bound: -0.0015587, upper bound: 0.0015610
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 2, lower bound: -0.0015587, upper bound: 0.0015611
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 2, lower bound: -0.0015942, upper bound: 0.0015992
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 2, lower bound: -0.0016023, upper bound: 0.0015899

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037366, 0.0037348
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026072, 0.0026060
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0045063, 0.0045085
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039597, 0.0039578
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0055212, 0.0055240
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0038218, 0.0038237
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015475, upper bound: 0.0015594
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015475, upper bound: 0.0015515
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037316, 0.0037405
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026041, 0.0026096
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0045129, 0.0045027
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039546, 0.0039636
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0055296, 0.0055166
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0038277, 0.0038186
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015503, upper bound: 0.0015526
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015503, upper bound: 0.0015528
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037968, 0.0038027
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026485, 0.0026485
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0045870, 0.0045802
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0040105, 0.0040165
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0056075, 0.0055989
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0038915, 0.0038854
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015931, upper bound: 0.0015979
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015888, upper bound: 0.0015982
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0038054, 0.0037937
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026485, 0.0026481
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0045766, 0.0045901
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0040193, 0.0040074
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0055943, 0.0056115
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0038822, 0.0038943
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015798, upper bound: 0.0015872
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015996, upper bound: 0.0015798
time: 1.83 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.47 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 2, lower bound: -0.0015475, upper bound: 0.0015594
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 2, lower bound: -0.0015475, upper bound: 0.0015515
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 2, lower bound: -0.0015503, upper bound: 0.0015526
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 2, lower bound: -0.0015503, upper bound: 0.0015528
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 2, lower bound: -0.0015931, upper bound: 0.0015979
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 2, lower bound: -0.0015888, upper bound: 0.0015982
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 2, lower bound: -0.0015798, upper bound: 0.0015872
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 2, lower bound: -0.0015996, upper bound: 0.0015798

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037067, 0.0037134
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025941, 0.0025983
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044835, 0.0044757
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039185, 0.0039253
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054762, 0.0054663
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037996, 0.0037927
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014705, upper bound: 0.0015272
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015139, upper bound: 0.0014759
time: 1.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037154, 0.0037048
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025996, 0.0025930
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044736, 0.0044859
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039274, 0.0039166
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054636, 0.0054792
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037907, 0.0038017
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015389, upper bound: 0.0015430
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015389, upper bound: 0.0015430
time: 1.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036637, 0.0036710
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025620, 0.0025665
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044324, 0.0044240
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038857, 0.0038931
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054280, 0.0054173
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037562, 0.0037487
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014858, upper bound: 0.0014912
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014858, upper bound: 0.0014875
time: 1.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036622, 0.0036746
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025610, 0.0025687
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044366, 0.0044222
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038842, 0.0038968
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054333, 0.0054151
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037599, 0.0037472
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014721, upper bound: 0.0014862
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014846, upper bound: 0.0014751
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037940, 0.0037998
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026482, 0.0026485
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0045836, 0.0045769
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0040078, 0.0040137
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0056034, 0.0055949
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0038885, 0.0038825
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015121, upper bound: 0.0015332
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015121, upper bound: 0.0015208
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037940, 0.0037999
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026482, 0.0026485
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0045837, 0.0045769
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0040078, 0.0040138
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0056036, 0.0055949
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0038886, 0.0038825
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015464, upper bound: 0.0015560
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015464, upper bound: 0.0015558
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037996, 0.0037918
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026485, 0.0026482
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0045753, 0.0045843
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0040144, 0.0040065
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0055929, 0.0056043
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0038803, 0.0038883
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014991, upper bound: 0.0015518
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014991, upper bound: 0.0015085
time: 2.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0038034, 0.0037879
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026485, 0.0026457
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0045708, 0.0045888
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0040183, 0.0040025
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0055871, 0.0056100
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0038763, 0.0038922
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015355, upper bound: 0.0015206
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015237, upper bound: 0.0015195
time: 1.89 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.67 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0014705, upper bound: 0.0015272
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0015139, upper bound: 0.0014759
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0015389, upper bound: 0.0015430
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0015389, upper bound: 0.0015430
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0014858, upper bound: 0.0014912
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0014858, upper bound: 0.0014875
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0014721, upper bound: 0.0014862
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0014846, upper bound: 0.0014751
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0015121, upper bound: 0.0015332
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0015121, upper bound: 0.0015208
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0015464, upper bound: 0.0015560
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0015464, upper bound: 0.0015558
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0014991, upper bound: 0.0015518
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0014991, upper bound: 0.0015085
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0015355, upper bound: 0.0015206
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 2, lower bound: -0.0015237, upper bound: 0.0015195

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0031506, 0.0032456
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0023267, 0.0023856
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0040010, 0.0038909
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0032813, 0.0033780
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0046908, 0.0045511
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0033122, 0.0032144
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014692, upper bound: 0.0015255
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014692, upper bound: 0.0015262
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036522, 0.0036370
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025604, 0.0025509
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0043950, 0.0044127
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038621, 0.0038466
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0053629, 0.0053854
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037210, 0.0037367
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015279, upper bound: 0.0015401
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015362, upper bound: 0.0015333
time: 2.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036476, 0.0036386
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025575, 0.0025519
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0043969, 0.0044073
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038574, 0.0038483
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0053654, 0.0053786
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037227, 0.0037319
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015374, upper bound: 0.0015411
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015374, upper bound: 0.0015416
time: 1.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037210, 0.0037546
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026120, 0.0026328
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0045347, 0.0044959
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039298, 0.0039640
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0055315, 0.0054822
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0038417, 0.0038072
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015040, upper bound: 0.0015251
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015040, upper bound: 0.0015252
time: 1.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037485, 0.0037268
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026290, 0.0026156
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0045026, 0.0045277
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039578, 0.0039357
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054907, 0.0055226
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0038132, 0.0038355
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015228, upper bound: 0.0015197
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015223, upper bound: 0.0015189
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037077, 0.0037079
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025946, 0.0025947
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044771, 0.0044769
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039196, 0.0039198
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054683, 0.0054680
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037939, 0.0037937
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014692, upper bound: 0.0015233
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014692, upper bound: 0.0014740
time: 1.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037020, 0.0037126
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025911, 0.0025977
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044826, 0.0044703
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039138, 0.0039246
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054752, 0.0054596
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037988, 0.0037878
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014692, upper bound: 0.0015233
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014692, upper bound: 0.0014740
time: 1.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0032528, 0.0033317
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0023903, 0.0024393
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0041021, 0.0040107
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0033838, 0.0034641
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0048156, 0.0046996
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0034009, 0.0033197
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014482, upper bound: 0.0014990
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014482, upper bound: 0.0014979
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036506, 0.0036564
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025564, 0.0025600
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044164, 0.0044097
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038714, 0.0038773
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054040, 0.0053955
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037411, 0.0037351
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014813, upper bound: 0.0014773
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014813, upper bound: 0.0014773
time: 1.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036719, 0.0036339
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025696, 0.0025461
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0043903, 0.0044343
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038931, 0.0038544
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0053709, 0.0054268
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037180, 0.0037570
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014451, upper bound: 0.0014840
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014451, upper bound: 0.0014392
time: 2.06 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.29 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0014692, upper bound: 0.0015255
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0014692, upper bound: 0.0015262
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0015279, upper bound: 0.0015401
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0015362, upper bound: 0.0015333
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0015374, upper bound: 0.0015411
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0015374, upper bound: 0.0015416
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0015040, upper bound: 0.0015251
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0015040, upper bound: 0.0015252
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0015228, upper bound: 0.0015197
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0015223, upper bound: 0.0015189
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0014692, upper bound: 0.0015233
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0014692, upper bound: 0.0014740
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0014692, upper bound: 0.0015233
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0014692, upper bound: 0.0014740
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0014482, upper bound: 0.0014990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0014482, upper bound: 0.0014979
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0014813, upper bound: 0.0014773
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0014813, upper bound: 0.0014773
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0014451, upper bound: 0.0014840
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.29
Output dim: 2, lower bound: -0.0014451, upper bound: 0.0014392

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0031478, 0.0032428
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0023248, 0.0023838
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0039976, 0.0038875
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0032785, 0.0033752
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0046867, 0.0045471
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0033093, 0.0032116
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014247, upper bound: 0.0014844
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014281, upper bound: 0.0014820
time: 1.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0031478, 0.0032429
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0023248, 0.0023838
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0039976, 0.0038875
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0032785, 0.0033752
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0046868, 0.0045470
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0033094, 0.0032115
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014663, upper bound: 0.0015249
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014663, upper bound: 0.0015247
time: 1.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036451, 0.0036336
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025571, 0.0025500
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0043920, 0.0044053
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038545, 0.0038428
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0053572, 0.0053742
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037175, 0.0037294
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014675, upper bound: 0.0014780
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014675, upper bound: 0.0014747
time: 1.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036480, 0.0036299
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025589, 0.0025477
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0043877, 0.0044087
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038575, 0.0038390
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0053517, 0.0053785
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037136, 0.0037323
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014728, upper bound: 0.0014728
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014846, upper bound: 0.0014714
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036455, 0.0036367
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025562, 0.0025508
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0043950, 0.0044051
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038552, 0.0038463
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0053625, 0.0053754
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037207, 0.0037297
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014612, upper bound: 0.0014744
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014719, upper bound: 0.0014650
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036457, 0.0036367
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025563, 0.0025507
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0043949, 0.0044053
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038554, 0.0038462
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0053624, 0.0053757
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037206, 0.0037299
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015362, upper bound: 0.0015404
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015362, upper bound: 0.0015406
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036526, 0.0036845
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025696, 0.0025894
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044536, 0.0044167
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038604, 0.0038929
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054291, 0.0053822
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037697, 0.0037369
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014975, upper bound: 0.0015225
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015054, upper bound: 0.0015128
time: 1.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036510, 0.0036863
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025685, 0.0025905
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044557, 0.0044147
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038587, 0.0038947
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054317, 0.0053798
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037715, 0.0037352
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014613, upper bound: 0.0014819
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014613, upper bound: 0.0014819
time: 2.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037466, 0.0037249
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026279, 0.0026144
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0045007, 0.0045259
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039559, 0.0039337
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054879, 0.0055198
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0038113, 0.0038336
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014465, upper bound: 0.0014867
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014300, upper bound: 0.0014360
time: 2.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0037466, 0.0037248
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026278, 0.0026143
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0045006, 0.0045259
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0039558, 0.0039336
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054877, 0.0055198
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0038112, 0.0038336
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015139, upper bound: 0.0015106
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015140, upper bound: 0.0015110
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0031516, 0.0032393
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0023272, 0.0023815
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0039934, 0.0038919
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0032824, 0.0033715
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0046815, 0.0045527
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0033056, 0.0032155
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014044, upper bound: 0.0014625
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014044, upper bound: 0.0014596
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0031460, 0.0032406
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0023237, 0.0023824
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0039950, 0.0038854
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0032766, 0.0033729
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0046835, 0.0045443
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0033071, 0.0032097
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014663, upper bound: 0.0015221
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014663, upper bound: 0.0015215
time: 1.82 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.67 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014247, upper bound: 0.0014844
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014281, upper bound: 0.0014820
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014663, upper bound: 0.0015249
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014663, upper bound: 0.0015247
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014675, upper bound: 0.0014780
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014675, upper bound: 0.0014747
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014728, upper bound: 0.0014728
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014846, upper bound: 0.0014714
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014612, upper bound: 0.0014744
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014719, upper bound: 0.0014650
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0015362, upper bound: 0.0015404
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0015362, upper bound: 0.0015406
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014975, upper bound: 0.0015225
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0015054, upper bound: 0.0015128
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014613, upper bound: 0.0014819
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014613, upper bound: 0.0014819
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014465, upper bound: 0.0014867
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014300, upper bound: 0.0014360
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0015139, upper bound: 0.0015106
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0015140, upper bound: 0.0015110
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014044, upper bound: 0.0014625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014044, upper bound: 0.0014596
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014663, upper bound: 0.0015221
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 2, lower bound: -0.0014663, upper bound: 0.0015215

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0031455, 0.0032408
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0023235, 0.0023826
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0039951, 0.0038847
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0032760, 0.0033730
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0046836, 0.0045435
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0033072, 0.0032092
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015160
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015164
time: 1.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0031457, 0.0032407
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0023236, 0.0023825
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0039950, 0.0038850
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0032762, 0.0033728
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0046834, 0.0045438
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0033071, 0.0032094
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015156
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015162
time: 1.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036429, 0.0036338
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025544, 0.0025488
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0043915, 0.0044021
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038527, 0.0038434
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0053583, 0.0053717
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037176, 0.0037270
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014600, upper bound: 0.0014738
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014703, upper bound: 0.0014640
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036429, 0.0036339
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025544, 0.0025489
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0043916, 0.0044020
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038526, 0.0038435
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0053585, 0.0053717
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037177, 0.0037269
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015071
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0014592
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0036479, 0.0036845
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0025676, 0.0025903
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0044541, 0.0044117
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0038569, 0.0038941
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0054300, 0.0053762
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0037697, 0.0037321
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014671, upper bound: 0.0014902
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014671, upper bound: 0.0014883
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0031438, 0.0032386
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0023224, 0.0023812
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0039925, 0.0038828
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0032743, 0.0033707
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0046803, 0.0045410
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0033049, 0.0032075
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015134
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015134
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823
1: -0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0031439, 0.0032383
2: 0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0023225, 0.0023810
3: -0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0039922, 0.0038829
4: -0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0032744, 0.0033704
5: 0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447
6: -0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0046799, 0.0045411
7: 0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0033047, 0.0032076
8: -0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088
9: 0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014539, upper bound: 0.0015191
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014638, upper bound: 0.0015105
time: 1.86 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015160
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015164
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015156
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015162
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0014600, upper bound: 0.0014738
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0014703, upper bound: 0.0014640
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015071
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0014592
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0014671, upper bound: 0.0014902
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0014671, upper bound: 0.0014883
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015134
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0014554, upper bound: 0.0015134
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0014539, upper bound: 0.0015191
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0014638, upper bound: 0.0015105

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.63 + 205.31 = 208.95 seconds
