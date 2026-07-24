## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00262656


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0063999, 0.0063999)
1: (-0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015947, 0.0015947)
2: (0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0084510, 0.0084510)
3: (-0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0038465, 0.0038465)
4: (0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0016357, 0.0016357)
5: (0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0106292, 0.0106292)
6: (-0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026978, 0.0026978)
7: (-0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0069800, 0.0069800)
8: (-0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0036707, 0.0036707)
9: (-0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0042564, 0.0042564)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.94 + 2.15 = 3.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0032832, upper bound: 0.0032832

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0031416, upper bound: 0.0031416
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0031416, upper bound: 0.0031416
time: 1.53 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.06 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.06
Output dim: 0, lower bound: -0.0031416, upper bound: 0.0031416
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.06
Output dim: 0, lower bound: -0.0031416, upper bound: 0.0031416

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0063778, 0.0063849
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015892, 0.0015909
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0084312, 0.0084218
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0038332, 0.0038375
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0016318, 0.0016300
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0106042, 0.0105924
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026885, 0.0026915
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0069559, 0.0069636
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0036580, 0.0036621
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0042464, 0.0042416

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0031239, upper bound: 0.0030537
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0030537, upper bound: 0.0031239
time: 1.14 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0063849, 0.0063999
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015909, 0.0015947
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0084510, 0.0084312
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0038375, 0.0038465
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0016357, 0.0016318
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0106292, 0.0106042
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026915, 0.0026978
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0069636, 0.0069800
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0036621, 0.0036707
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0042564, 0.0042464

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0031032, upper bound: 0.0030968
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0030968, upper bound: 0.0031032
time: 1.94 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.84 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.84
Output dim: 0, lower bound: -0.0031239, upper bound: 0.0030537
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.84
Output dim: 0, lower bound: -0.0030537, upper bound: 0.0031239
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.84
Output dim: 0, lower bound: -0.0031032, upper bound: 0.0030968
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.84
Output dim: 0, lower bound: -0.0030968, upper bound: 0.0031032

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0057848, 0.0057618
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014414, 0.0014357
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0076084, 0.0076388
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0034768, 0.0034630
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014726, 0.0014785
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0095694, 0.0096076
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0024385, 0.0024288
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0063091, 0.0062841
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0033179, 0.0033047
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0038320, 0.0038473

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0030445, upper bound: 0.0029080
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029813, upper bound: 0.0029786
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0057547, 0.0057903
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014339, 0.0014428
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0076460, 0.0075990
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0034587, 0.0034801
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014799, 0.0014708
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0096167, 0.0095576
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0024258, 0.0024408
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0062763, 0.0063151
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0033006, 0.0033211
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0038509, 0.0038273

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029697, upper bound: 0.0030375
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029746, upper bound: 0.0030288
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0062846, 0.0062928
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015659, 0.0015680
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0083095, 0.0082987
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0037772, 0.0037821
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0016083, 0.0016062
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0104512, 0.0104376
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026492, 0.0026526
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0068542, 0.0068632
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0036046, 0.0036093
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0041851, 0.0041797

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0030668, upper bound: 0.0030116
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0030133, upper bound: 0.0030591
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0062780, 0.0062994
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015643, 0.0015696
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0083183, 0.0082900
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0037732, 0.0037861
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0016100, 0.0016045
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0104622, 0.0104266
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026464, 0.0026554
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0068470, 0.0068704
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0036008, 0.0036131
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0041895, 0.0041753

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0030624, upper bound: 0.0030709
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0030602, upper bound: 0.0030710
time: 1.27 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.41 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -0.0030445, upper bound: 0.0029080
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -0.0029813, upper bound: 0.0029786
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -0.0029697, upper bound: 0.0030375
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -0.0029746, upper bound: 0.0030288
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -0.0030668, upper bound: 0.0030116
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -0.0030133, upper bound: 0.0030591
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -0.0030624, upper bound: 0.0030709
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -0.0030602, upper bound: 0.0030710

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0055420, 0.0054652
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013809, 0.0013618
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0072167, 0.0073181
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0033309, 0.0032847
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013968, 0.0014164
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0090768, 0.0092043
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023361, 0.0023038
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0060443, 0.0059606
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031786, 0.0031346
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036347, 0.0036858

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0024714, upper bound: 0.0023693
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0024714, upper bound: 0.0023693
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0054882, 0.0055186
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013675, 0.0013751
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0072873, 0.0072471
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032986, 0.0033168
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014104, 0.0014027
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0091654, 0.0091149
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023135, 0.0023263
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0059857, 0.0060188
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031478, 0.0031652
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036702, 0.0036500

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029281, upper bound: 0.0028588
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028534, upper bound: 0.0029239
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0057416, 0.0057828
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014306, 0.0014409
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0076362, 0.0075817
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0034508, 0.0034756
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014780, 0.0014674
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0096043, 0.0095357
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0024203, 0.0024377
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0062620, 0.0063070
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0032931, 0.0033168
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0038460, 0.0038185

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028931, upper bound: 0.0028996
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028310, upper bound: 0.0029578
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0057471, 0.0057771
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014320, 0.0014395
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0076287, 0.0075889
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0034541, 0.0034722
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014765, 0.0014688
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0095948, 0.0095449
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0024226, 0.0024353
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0062680, 0.0063008
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0032963, 0.0033135
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0038422, 0.0038222

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029405, upper bound: 0.0029445
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028830, upper bound: 0.0029927
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0061737, 0.0061624
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015383, 0.0015355
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0081373, 0.0081522
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0037105, 0.0037038
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015750, 0.0015779
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0102346, 0.0102534
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026024, 0.0025977
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0067332, 0.0067209
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0035409, 0.0035345
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0040984, 0.0041059

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029726, upper bound: 0.0029323
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029803, upper bound: 0.0029244
time: 1.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0061548, 0.0061793
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015336, 0.0015397
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0081597, 0.0081273
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0036992, 0.0037140
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015793, 0.0015730
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0102628, 0.0102221
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0025945, 0.0026048
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0067127, 0.0067394
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0035301, 0.0035442
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0041097, 0.0040934

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029469, upper bound: 0.0029473
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028996, upper bound: 0.0029879
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0062290, 0.0062542
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015521, 0.0015584
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0082586, 0.0082254
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0037438, 0.0037589
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015984, 0.0015920
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0103871, 0.0103454
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026258, 0.0026364
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0067937, 0.0068211
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0035727, 0.0035871
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0041595, 0.0041427

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029932, upper bound: 0.0029508
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029516, upper bound: 0.0030040
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0062299, 0.0062505
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015523, 0.0015575
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0082537, 0.0082265
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0037443, 0.0037567
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015975, 0.0015922
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0103810, 0.0103467
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0026261, 0.0026348
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0067945, 0.0068170
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0035732, 0.0035850
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0041570, 0.0041433

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029812, upper bound: 0.0029289
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029215, upper bound: 0.0029914
time: 1.67 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.37 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0024714, upper bound: 0.0023693
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0024714, upper bound: 0.0023693
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0029281, upper bound: 0.0028588
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0028534, upper bound: 0.0029239
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0028931, upper bound: 0.0028996
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0028310, upper bound: 0.0029578
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0029405, upper bound: 0.0029445
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0028830, upper bound: 0.0029927
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0029726, upper bound: 0.0029323
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0029803, upper bound: 0.0029244
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0029469, upper bound: 0.0029473
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0028996, upper bound: 0.0029879
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0029932, upper bound: 0.0029508
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0029516, upper bound: 0.0030040
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0029812, upper bound: 0.0029289
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.37
Output dim: 0, lower bound: -0.0029215, upper bound: 0.0029914

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0051819, 0.0051443
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012912, 0.0012818
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0067929, 0.0068426
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031145, 0.0030918
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013148, 0.0013244
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0085437, 0.0086062
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021843, 0.0021685
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0056516, 0.0056105
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029721, 0.0029505
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034213, 0.0034463

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028449, upper bound: 0.0027783
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028508, upper bound: 0.0027742
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0051138, 0.0052055
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012742, 0.0012971
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0068738, 0.0067528
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0030736, 0.0031286
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013304, 0.0013070
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0086454, 0.0084932
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021557, 0.0021943
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0055774, 0.0056773
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029331, 0.0029856
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034620, 0.0034011

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028169, upper bound: 0.0028722
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028169, upper bound: 0.0028834
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0054988, 0.0054870
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013701, 0.0013672
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0072455, 0.0072611
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0033049, 0.0032978
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014023, 0.0014054
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0091129, 0.0091325
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023179, 0.0023129
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0059972, 0.0059843
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031539, 0.0031471
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036492, 0.0036571

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028211, upper bound: 0.0027787
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027466, upper bound: 0.0028225
time: 1.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0054457, 0.0055395
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013569, 0.0013803
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0073149, 0.0071910
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032730, 0.0033294
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014158, 0.0013918
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0092002, 0.0090443
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022955, 0.0023351
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0059393, 0.0060417
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031234, 0.0031773
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036842, 0.0036217

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0020838, upper bound: 0.0021186
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0020838, upper bound: 0.0021186
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0056264, 0.0056362
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014020, 0.0014044
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0074426, 0.0074296
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0033816, 0.0033875
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014405, 0.0014380
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0093608, 0.0093445
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023717, 0.0023759
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0061364, 0.0061471
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0032271, 0.0032327
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0037485, 0.0037420

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028907, upper bound: 0.0028396
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028177, upper bound: 0.0028946
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0056062, 0.0056521
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013969, 0.0014084
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0074636, 0.0074029
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0033695, 0.0033971
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014446, 0.0014328
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0093872, 0.0093109
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023632, 0.0023826
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0061143, 0.0061645
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0032155, 0.0032418
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0037591, 0.0037285

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028325, upper bound: 0.0028676
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027833, upper bound: 0.0029441
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0061599, 0.0061545
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015349, 0.0015335
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0081269, 0.0081341
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0037023, 0.0036990
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015729, 0.0015743
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0102215, 0.0102306
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0025966, 0.0025943
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0067183, 0.0067123
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0035331, 0.0035299
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0040931, 0.0040968

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0029235, upper bound: 0.0028337
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028509, upper bound: 0.0028819
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0061656, 0.0061487
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015363, 0.0015321
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0081192, 0.0081416
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0037057, 0.0036955
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015715, 0.0015758
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0102119, 0.0102400
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0025990, 0.0025919
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0067245, 0.0067060
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0035363, 0.0035266
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0040893, 0.0041006

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0021238, upper bound: 0.0021096
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0021238, upper bound: 0.0021096
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0060313, 0.0060154
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015028, 0.0014989
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0079433, 0.0079643
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0036250, 0.0036154
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015374, 0.0015415
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0099906, 0.0100170
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0025424, 0.0025357
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0065780, 0.0065607
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0034593, 0.0034502
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0040007, 0.0040112

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028387, upper bound: 0.0027463
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027525, upper bound: 0.0028452
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0059907, 0.0060492
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014927, 0.0015073
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0079879, 0.0079107
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0036006, 0.0036357
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015460, 0.0015311
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0100466, 0.0099496
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0025253, 0.0025499
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0065337, 0.0065975
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0034360, 0.0034695
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0040231, 0.0039842

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0023092, upper bound: 0.0023613
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0023092, upper bound: 0.0023613
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0060997, 0.0060908
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015199, 0.0015177
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0080428, 0.0080546
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0036661, 0.0036607
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015567, 0.0015589
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0101157, 0.0101306
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0025712, 0.0025675
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0066526, 0.0066428
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0034985, 0.0034934
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0040508, 0.0040567

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028912, upper bound: 0.0027508
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027896, upper bound: 0.0028467
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0060657, 0.0061301
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0015114, 0.0015275
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0080947, 0.0080097
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0036457, 0.0036844
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015667, 0.0015503
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0101810, 0.0100741
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0025569, 0.0025841
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0066155, 0.0066857
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0034790, 0.0035160
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0040769, 0.0040341

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028981, upper bound: 0.0028735
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028425, upper bound: 0.0029515
time: 1.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0059836, 0.0059541
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014909, 0.0014836
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0078623, 0.0079012
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0035963, 0.0035786
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015217, 0.0015293
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0098887, 0.0099377
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0025223, 0.0025099
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0065259, 0.0064938
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0034319, 0.0034150
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0039599, 0.0039795

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028837, upper bound: 0.0028462
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028923, upper bound: 0.0028417
time: 1.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0059334, 0.0060075
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014784, 0.0014969
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0079328, 0.0078349
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0035661, 0.0036107
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015354, 0.0015164
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0099774, 0.0098543
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0025011, 0.0025324
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0064712, 0.0065520
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0034031, 0.0034456
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0039954, 0.0039461

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028342, upper bound: 0.0029031
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028395, upper bound: 0.0028963
time: 1.42 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 7.41 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028449, upper bound: 0.0027783
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028508, upper bound: 0.0027742
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028169, upper bound: 0.0028722
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028169, upper bound: 0.0028834
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028211, upper bound: 0.0027787
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0027466, upper bound: 0.0028225
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0020838, upper bound: 0.0021186
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0020838, upper bound: 0.0021186
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028907, upper bound: 0.0028396
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028177, upper bound: 0.0028946
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028325, upper bound: 0.0028676
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0027833, upper bound: 0.0029441
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0029235, upper bound: 0.0028337
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028509, upper bound: 0.0028819
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0021238, upper bound: 0.0021096
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0021238, upper bound: 0.0021096
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028387, upper bound: 0.0027463
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0027525, upper bound: 0.0028452
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0023092, upper bound: 0.0023613
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0023092, upper bound: 0.0023613
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028912, upper bound: 0.0027508
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0027896, upper bound: 0.0028467
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028981, upper bound: 0.0028735
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028425, upper bound: 0.0029515
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028837, upper bound: 0.0028462
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028923, upper bound: 0.0028417
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028342, upper bound: 0.0029031
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.41
Output dim: 0, lower bound: -0.0028395, upper bound: 0.0028963

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0051688, 0.0051364
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012879, 0.0012799
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0067826, 0.0068254
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031066, 0.0030872
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013128, 0.0013210
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0085307, 0.0085845
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021788, 0.0021652
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0056373, 0.0056020
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029646, 0.0029460
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034161, 0.0034376

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028069, upper bound: 0.0027382
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028016, upper bound: 0.0027392
time: 1.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0051737, 0.0051312
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012892, 0.0012786
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0067757, 0.0068318
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031096, 0.0030840
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013114, 0.0013223
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0085220, 0.0085927
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021809, 0.0021630
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0056427, 0.0055963
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029674, 0.0029430
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034126, 0.0034409

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028121, upper bound: 0.0027003
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027781, upper bound: 0.0027374
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0049937, 0.0050683
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012443, 0.0012629
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0066926, 0.0065941
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0030014, 0.0030462
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012953, 0.0012763
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0084175, 0.0082937
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021050, 0.0021365
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0054463, 0.0055277
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028642, 0.0029069
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033707, 0.0033211

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027447, upper bound: 0.0027364
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027161, upper bound: 0.0028081
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0049916, 0.0050853
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012438, 0.0012671
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0067151, 0.0065914
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0030001, 0.0030564
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012997, 0.0012757
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0084458, 0.0082902
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021041, 0.0021436
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0054441, 0.0055462
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028630, 0.0029167
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033821, 0.0033198

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027355, upper bound: 0.0028055
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027424, upper bound: 0.0028021
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053738, 0.0053176
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013390, 0.0013250
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0070218, 0.0070960
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032298, 0.0031960
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013591, 0.0013734
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0088316, 0.0089250
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022652, 0.0022416
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0058609, 0.0057996
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030822, 0.0030500
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035366, 0.0035739

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0020121, upper bound: 0.0020020
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0020121, upper bound: 0.0020020
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053294, 0.0053479
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013279, 0.0013325
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0070618, 0.0070374
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032031, 0.0032142
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013668, 0.0013621
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0088819, 0.0088512
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022465, 0.0022543
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0058125, 0.0058326
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030567, 0.0030673
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035567, 0.0035444

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027119, upper bound: 0.0027524
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026545, upper bound: 0.0027834
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053984, 0.0053402
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013451, 0.0013306
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0070516, 0.0071285
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032446, 0.0032096
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013648, 0.0013797
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0088691, 0.0089657
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022756, 0.0022511
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0058877, 0.0058242
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030963, 0.0030629
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035516, 0.0035903

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0020742, upper bound: 0.0020420
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0020742, upper bound: 0.0020420
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053303, 0.0054063
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013282, 0.0013471
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0071390, 0.0070386
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032037, 0.0032493
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013817, 0.0013623
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0089789, 0.0088527
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022469, 0.0022790
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0058135, 0.0058963
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030572, 0.0031008
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035956, 0.0035450

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027779, upper bound: 0.0028558
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027755, upper bound: 0.0028577
time: 1.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053759, 0.0053561
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013395, 0.0013346
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0070726, 0.0070988
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032311, 0.0032191
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013689, 0.0013740
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0088955, 0.0089284
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022661, 0.0022578
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0058631, 0.0058415
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030834, 0.0030720
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035621, 0.0035753

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027982, upper bound: 0.0028303
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027981, upper bound: 0.0028309
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053101, 0.0054261
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013231, 0.0013520
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0071651, 0.0070119
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031915, 0.0032613
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013868, 0.0013571
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0090119, 0.0088191
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022384, 0.0022873
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0057914, 0.0059180
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030456, 0.0031122
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036087, 0.0035316

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026744, upper bound: 0.0027349
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0028452
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0059190, 0.0058472
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014749, 0.0014570
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0077212, 0.0078160
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0035575, 0.0035144
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014944, 0.0015128
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0097112, 0.0098305
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0024951, 0.0024648
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0064556, 0.0063772
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0033949, 0.0033537
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0038888, 0.0039366

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028257, upper bound: 0.0026345
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027139, upper bound: 0.0027279
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0058529, 0.0059007
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014584, 0.0014703
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0077918, 0.0077286
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0035177, 0.0035465
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015081, 0.0014959
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0098001, 0.0097206
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0024672, 0.0024874
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0063834, 0.0064356
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0033570, 0.0033844
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0039244, 0.0038926

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027715, upper bound: 0.0027646
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027450, upper bound: 0.0028062
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0055952, 0.0055212
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013942, 0.0013757
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0072906, 0.0073884
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0033629, 0.0033184
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014111, 0.0014300
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0091697, 0.0092927
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023586, 0.0023274
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0061024, 0.0060216
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0032092, 0.0031667
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036719, 0.0037212

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027855, upper bound: 0.0026380
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027356, upper bound: 0.0026934
time: 1.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0055366, 0.0055923
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013796, 0.0013934
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0073845, 0.0073111
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0033277, 0.0033611
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014293, 0.0014150
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0092878, 0.0091954
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023339, 0.0023573
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0060385, 0.0060991
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031756, 0.0032075
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0037192, 0.0036822

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026529, upper bound: 0.0027534
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026605, upper bound: 0.0027511
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0056734, 0.0056018
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014137, 0.0013958
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0073971, 0.0074916
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0034099, 0.0033668
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014317, 0.0014500
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0093036, 0.0094225
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023915, 0.0023614
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0061876, 0.0061096
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0032540, 0.0032130
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0037256, 0.0037732

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028387, upper bound: 0.0026362
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027720, upper bound: 0.0026981
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0056102, 0.0056608
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013979, 0.0014105
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0074750, 0.0074083
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0033719, 0.0034023
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014468, 0.0014339
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0094016, 0.0093176
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023649, 0.0023862
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0061188, 0.0061739
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0032178, 0.0032468
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0037648, 0.0037312

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027711, upper bound: 0.0027350
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026959, upper bound: 0.0028287
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0058019, 0.0058118
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014457, 0.0014481
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0076744, 0.0076613
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0034871, 0.0034930
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014854, 0.0014828
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0096524, 0.0096359
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0024457, 0.0024499
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0063278, 0.0063386
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0033277, 0.0033334
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0038652, 0.0038586

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028183, upper bound: 0.0027249
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027604, upper bound: 0.0027962
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0057476, 0.0058818
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014322, 0.0014656
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0077668, 0.0075897
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0034545, 0.0035351
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015033, 0.0014690
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0097686, 0.0095458
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0024228, 0.0024794
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0062686, 0.0064149
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0032966, 0.0033735
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0039118, 0.0038226

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027665, upper bound: 0.0028102
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027006, upper bound: 0.0028731
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0059706, 0.0059466
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014877, 0.0014817
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0078524, 0.0078841
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0035885, 0.0035741
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015198, 0.0015260
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0098763, 0.0099161
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0025168, 0.0025067
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0065118, 0.0064856
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0034245, 0.0034107
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0039549, 0.0039708

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028457, upper bound: 0.0027739
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028081, upper bound: 0.0028079
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0059758, 0.0059411
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014890, 0.0014804
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0078452, 0.0078910
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0035916, 0.0035708
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015184, 0.0015273
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0098671, 0.0099248
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0025190, 0.0025044
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0065175, 0.0064796
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0034275, 0.0034076
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0039512, 0.0039743

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028428, upper bound: 0.0027174
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027829, upper bound: 0.0027924
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0059204, 0.0059997
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014752, 0.0014950
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0079226, 0.0078178
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0035583, 0.0036060
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015334, 0.0015131
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0099645, 0.0098328
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0024957, 0.0025291
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0064570, 0.0065436
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0033957, 0.0034412
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0039902, 0.0039375

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 156

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028156, upper bound: 0.0028228
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027444, upper bound: 0.0028842
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0059263, 0.0059945
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014767, 0.0014937
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0079157, 0.0078256
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0035619, 0.0036029
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0015321, 0.0015146
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0099558, 0.0098426
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0024982, 0.0025269
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0064635, 0.0065378
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0033991, 0.0034382
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0039868, 0.0039414

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0020153, upper bound: 0.0020320
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0020153, upper bound: 0.0020320
time: 1.10 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.04 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0028069, upper bound: 0.0027382
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0028016, upper bound: 0.0027392
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0028121, upper bound: 0.0027003
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027781, upper bound: 0.0027374
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027447, upper bound: 0.0027364
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027161, upper bound: 0.0028081
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027355, upper bound: 0.0028055
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027424, upper bound: 0.0028021
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0020121, upper bound: 0.0020020
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0020121, upper bound: 0.0020020
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027119, upper bound: 0.0027524
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0026545, upper bound: 0.0027834
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0020742, upper bound: 0.0020420
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0020742, upper bound: 0.0020420
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027779, upper bound: 0.0028558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027755, upper bound: 0.0028577
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027982, upper bound: 0.0028303
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027981, upper bound: 0.0028309
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0026744, upper bound: 0.0027349
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0025881, upper bound: 0.0028452
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0028257, upper bound: 0.0026345
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027139, upper bound: 0.0027279
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027715, upper bound: 0.0027646
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027450, upper bound: 0.0028062
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027855, upper bound: 0.0026380
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027356, upper bound: 0.0026934
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0026529, upper bound: 0.0027534
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0026605, upper bound: 0.0027511
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0028387, upper bound: 0.0026362
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027720, upper bound: 0.0026981
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027711, upper bound: 0.0027350
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0026959, upper bound: 0.0028287
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0028183, upper bound: 0.0027249
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027604, upper bound: 0.0027962
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027665, upper bound: 0.0028102
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027006, upper bound: 0.0028731
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0028457, upper bound: 0.0027739
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0028081, upper bound: 0.0028079
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0028428, upper bound: 0.0027174
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027829, upper bound: 0.0027924
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0028156, upper bound: 0.0028228
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0027444, upper bound: 0.0028842
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0020153, upper bound: 0.0020320
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0020153, upper bound: 0.0020320

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0050484, 0.0050062
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012579, 0.0012474
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0066106, 0.0066664
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0030343, 0.0030089
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012795, 0.0012903
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0083144, 0.0083846
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021281, 0.0021103
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0055060, 0.0054600
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028956, 0.0028713
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033295, 0.0033575

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027093, upper bound: 0.0025347
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026022, upper bound: 0.0026382
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0050362, 0.0050160
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012549, 0.0012499
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0066236, 0.0066502
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0030269, 0.0030148
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012820, 0.0012871
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0083308, 0.0083643
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021229, 0.0021144
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0054927, 0.0054707
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028886, 0.0028770
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033360, 0.0033494

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027254, upper bound: 0.0026111
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026855, upper bound: 0.0026607
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0050747, 0.0050110
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012645, 0.0012486
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0066170, 0.0067011
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0030500, 0.0030118
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012807, 0.0012970
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0083224, 0.0084282
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021392, 0.0021123
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0055347, 0.0054652
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029106, 0.0028741
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033327, 0.0033750

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027727, upper bound: 0.0026617
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027677, upper bound: 0.0026628
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0050535, 0.0050273
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012592, 0.0012527
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0066385, 0.0066731
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0030373, 0.0030216
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012849, 0.0012916
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0083495, 0.0083930
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021302, 0.0021192
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0055116, 0.0054830
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028985, 0.0028835
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033435, 0.0033609

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027040, upper bound: 0.0026104
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026574, upper bound: 0.0026613
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0048594, 0.0049045
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012108, 0.0012221
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0064763, 0.0064167
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029206, 0.0029477
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012535, 0.0012419
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0081455, 0.0080706
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020484, 0.0020674
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0052998, 0.0053490
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0027871, 0.0028130
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0032618, 0.0032318

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0023382, upper bound: 0.0023501
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0023349, upper bound: 0.0023520
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0048299, 0.0049479
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012035, 0.0012329
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0065337, 0.0063778
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029029, 0.0029739
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012646, 0.0012344
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0082177, 0.0080216
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020360, 0.0020857
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0052677, 0.0053964
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0027702, 0.0028379
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0032907, 0.0032122

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026282, upper bound: 0.0027157
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026317, upper bound: 0.0027139
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0049783, 0.0050765
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012405, 0.0012649
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0067035, 0.0065738
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029921, 0.0030512
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012975, 0.0012723
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0084313, 0.0082681
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020985, 0.0021399
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0054296, 0.0055367
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028554, 0.0029117
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033762, 0.0033109

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026984, upper bound: 0.0027736
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026970, upper bound: 0.0027736
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0049841, 0.0050720
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012419, 0.0012638
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0066976, 0.0065814
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029956, 0.0030484
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012963, 0.0012738
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0084238, 0.0082777
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021010, 0.0021380
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0054358, 0.0055318
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028586, 0.0029091
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033732, 0.0033147

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026382, upper bound: 0.0025931
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025441, upper bound: 0.0027035
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0052350, 0.0052347
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013044, 0.0013043
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0069123, 0.0069128
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031464, 0.0031462
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013379, 0.0013380
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0086939, 0.0086945
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022068, 0.0022066
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0057096, 0.0057092
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030026, 0.0030024
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034814, 0.0034817

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026791, upper bound: 0.0027188
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026791, upper bound: 0.0027189
time: 1.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0052162, 0.0052491
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012997, 0.0013079
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0069314, 0.0068880
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031351, 0.0031549
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013416, 0.0013332
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0087179, 0.0086633
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021988, 0.0022127
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0056890, 0.0057249
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029918, 0.0030107
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034910, 0.0034692

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026147, upper bound: 0.0027354
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026132, upper bound: 0.0027414
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0052024, 0.0052672
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012963, 0.0013124
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0069553, 0.0068697
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031268, 0.0031657
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013462, 0.0013296
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0087479, 0.0086403
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021930, 0.0022203
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0056740, 0.0057446
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029839, 0.0030211
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035031, 0.0034600

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027426, upper bound: 0.0028202
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027417, upper bound: 0.0028209
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0051951, 0.0052784
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012945, 0.0013152
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0069701, 0.0068600
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031224, 0.0031725
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013490, 0.0013277
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0087665, 0.0086281
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021899, 0.0022250
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0056659, 0.0057568
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029797, 0.0030275
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035105, 0.0034551

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027003, upper bound: 0.0027354
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026365, upper bound: 0.0027798
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053302, 0.0053122
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013281, 0.0013237
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0070147, 0.0070385
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032036, 0.0031928
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013577, 0.0013623
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0088227, 0.0088526
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022469, 0.0022393
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0058134, 0.0057937
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030572, 0.0030469
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035330, 0.0035450

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026893, upper bound: 0.0026262
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026056, upper bound: 0.0027243
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053332, 0.0053104
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013289, 0.0013232
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0070123, 0.0070425
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032054, 0.0031917
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013572, 0.0013631
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0088197, 0.0088576
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022481, 0.0022385
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0058166, 0.0057917
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030589, 0.0030458
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035318, 0.0035470

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0020003, upper bound: 0.0019924
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0020003, upper bound: 0.0019924
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0048570, 0.0049083
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012102, 0.0012230
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0064814, 0.0064136
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029192, 0.0029501
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012545, 0.0012413
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0081519, 0.0080666
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020474, 0.0020690
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0052972, 0.0053532
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0027858, 0.0028152
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0032644, 0.0032302

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025968, upper bound: 0.0026020
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025430, upper bound: 0.0026559
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0047923, 0.0049761
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0011941, 0.0012399
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0065709, 0.0063281
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0028803, 0.0029908
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012718, 0.0012248
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0082644, 0.0079591
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020201, 0.0020976
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0052266, 0.0054271
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0027486, 0.0028541
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033094, 0.0031872

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025107, upper bound: 0.0027219
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0024602, upper bound: 0.0027699
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0054737, 0.0053376
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013639, 0.0013300
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0070482, 0.0072279
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032898, 0.0032080
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013642, 0.0013990
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0088648, 0.0090908
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023074, 0.0022500
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0059698, 0.0058214
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031395, 0.0030614
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035498, 0.0036404

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0019280, upper bound: 0.0018399
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0019280, upper bound: 0.0018399
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0054090, 0.0054029
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013478, 0.0013463
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0071345, 0.0071426
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032510, 0.0032473
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013809, 0.0013824
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0089734, 0.0089835
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022801, 0.0022775
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0058993, 0.0058927
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031024, 0.0030989
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035933, 0.0035974

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026353, upper bound: 0.0025932
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025820, upper bound: 0.0026510
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0057298, 0.0057374
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014277, 0.0014296
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0075761, 0.0075661
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0034438, 0.0034483
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014663, 0.0014644
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0095288, 0.0095162
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0024153, 0.0024185
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0062491, 0.0062574
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0032864, 0.0032907
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0038157, 0.0038107

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026974, upper bound: 0.0026389
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026334, upper bound: 0.0026841
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0056898, 0.0057774
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014178, 0.0014396
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0076289, 0.0075134
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0034198, 0.0034724
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014766, 0.0014542
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0095952, 0.0094498
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023985, 0.0024354
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0062056, 0.0063010
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0032634, 0.0033136
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0038423, 0.0037841

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026433, upper bound: 0.0026057
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025412, upper bound: 0.0026997
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053433, 0.0052011
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013314, 0.0012960
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0068680, 0.0070558
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032115, 0.0031260
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013293, 0.0013656
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0086381, 0.0088743
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022524, 0.0021924
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0058276, 0.0056725
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030647, 0.0029831
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034591, 0.0035537

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027497, upper bound: 0.0025955
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027485, upper bound: 0.0026008
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0052759, 0.0052582
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013146, 0.0013102
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0069434, 0.0069668
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031710, 0.0031603
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013439, 0.0013484
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0087329, 0.0087624
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022240, 0.0022165
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0057541, 0.0057348
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030260, 0.0030159
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034970, 0.0035088

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0022755, upper bound: 0.0022680
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0022660, upper bound: 0.0022698
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0055229, 0.0055841
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013762, 0.0013914
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0073737, 0.0072929
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0033194, 0.0033562
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014272, 0.0014115
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0092742, 0.0091726
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023281, 0.0023539
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0060235, 0.0060902
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031677, 0.0032028
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0037138, 0.0036731

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025735, upper bound: 0.0026204
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025263, upper bound: 0.0026744
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0055279, 0.0055785
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013774, 0.0013900
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0073664, 0.0072996
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0033224, 0.0033529
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014258, 0.0014128
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0092650, 0.0091809
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023302, 0.0023516
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0060290, 0.0060842
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031706, 0.0031996
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0037101, 0.0036765

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026297, upper bound: 0.0027163
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026297, upper bound: 0.0027167
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053956, 0.0052632
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013444, 0.0013114
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0069500, 0.0071249
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032429, 0.0031633
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013452, 0.0013790
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0087413, 0.0089612
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022744, 0.0022186
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0058847, 0.0057403
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030947, 0.0030187
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035004, 0.0035885

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028203, upper bound: 0.0025396
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027511, upper bound: 0.0026181
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053354, 0.0053279
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013294, 0.0013276
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0070354, 0.0070453
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032067, 0.0032022
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013617, 0.0013636
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0088487, 0.0088611
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022490, 0.0022459
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0058190, 0.0058108
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030601, 0.0030558
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035434, 0.0035484

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026749, upper bound: 0.0026102
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026794, upper bound: 0.0026029
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0050140, 0.0050396
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012494, 0.0012557
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0066548, 0.0066209
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0030136, 0.0030290
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012880, 0.0012815
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0083700, 0.0083274
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021136, 0.0021244
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0054685, 0.0054964
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028758, 0.0028905
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033517, 0.0033346

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0021266, upper bound: 0.0021121
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0021266, upper bound: 0.0021121
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0049893, 0.0050706
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012432, 0.0012635
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0066957, 0.0065884
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029987, 0.0030476
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012959, 0.0012752
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0084214, 0.0082864
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021032, 0.0021374
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0054416, 0.0055302
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028617, 0.0029083
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033723, 0.0033182

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0022908, upper bound: 0.0023774
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0022903, upper bound: 0.0023845
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0055308, 0.0054913
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013781, 0.0013683
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0072512, 0.0073033
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0033242, 0.0033004
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014035, 0.0014135
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0091201, 0.0091857
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023314, 0.0023148
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0060321, 0.0059890
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031722, 0.0031496
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036521, 0.0036783

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0024029, upper bound: 0.0023212
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0023984, upper bound: 0.0023239
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0054808, 0.0055465
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013657, 0.0013820
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0073241, 0.0072374
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032941, 0.0033336
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014176, 0.0014008
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0092118, 0.0091027
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023104, 0.0023381
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0059776, 0.0060493
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031436, 0.0031813
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036888, 0.0036451

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0021189, upper bound: 0.0021654
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0021189, upper bound: 0.0021654
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0054767, 0.0055613
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013646, 0.0013857
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0073436, 0.0072319
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032917, 0.0033425
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014213, 0.0013997
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0092363, 0.0090959
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023086, 0.0023443
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0059731, 0.0060654
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031412, 0.0031897
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036986, 0.0036424

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026727, upper bound: 0.0027228
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026781, upper bound: 0.0027212
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0054266, 0.0056108
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013522, 0.0013981
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0074091, 0.0071657
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032615, 0.0033723
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014340, 0.0013869
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0093186, 0.0090126
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022875, 0.0023652
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0059185, 0.0061194
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031125, 0.0032181
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0037316, 0.0036090

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025941, upper bound: 0.0026681
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025000, upper bound: 0.0027700
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0058853, 0.0058440
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014665, 0.0014562
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0077169, 0.0077715
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0035372, 0.0035124
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014936, 0.0015042
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0097058, 0.0097745
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0024809, 0.0024634
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0064188, 0.0063737
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0033756, 0.0033519
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0038866, 0.0039141

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027479, upper bound: 0.0025752
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026391, upper bound: 0.0026660
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0058680, 0.0058630
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014621, 0.0014609
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0077420, 0.0077486
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0035268, 0.0035238
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014985, 0.0014997
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0097374, 0.0097457
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0024736, 0.0024715
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0063999, 0.0063944
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0033656, 0.0033628
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0038993, 0.0039026

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027576, upper bound: 0.0026834
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027086, upper bound: 0.0027589
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0056804, 0.0055917
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014154, 0.0013933
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0073838, 0.0075010
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0034141, 0.0033608
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014291, 0.0014518
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0092869, 0.0094342
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023945, 0.0023571
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0061953, 0.0060985
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0032581, 0.0032072
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0037189, 0.0037779

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0028050, upper bound: 0.0026513
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027644, upper bound: 0.0026778
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0056264, 0.0056555
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0014019, 0.0014092
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0074681, 0.0074296
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0033816, 0.0033991
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014454, 0.0014380
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0093929, 0.0093444
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023717, 0.0023840
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0061364, 0.0061681
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0032270, 0.0032438
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0037613, 0.0037419

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0019481, upper bound: 0.0019642
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0019481, upper bound: 0.0019642
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053437, 0.0053906
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013315, 0.0013432
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0071182, 0.0070564
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032117, 0.0032399
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013777, 0.0013657
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0089528, 0.0088750
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022526, 0.0022723
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0058281, 0.0058792
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030649, 0.0030918
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035851, 0.0035540

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0019963, upper bound: 0.0019985
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0019963, upper bound: 0.0019985
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053114, 0.0054187
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013235, 0.0013502
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0071553, 0.0070137
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031923, 0.0032568
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013849, 0.0013575
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0089995, 0.0088214
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022390, 0.0022842
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0057929, 0.0059099
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030464, 0.0031079
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036038, 0.0035325

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027076, upper bound: 0.0028006
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026668, upper bound: 0.0028479
time: 1.25 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.54 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027093, upper bound: 0.0025347
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026022, upper bound: 0.0026382
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027254, upper bound: 0.0026111
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026855, upper bound: 0.0026607
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027727, upper bound: 0.0026617
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027677, upper bound: 0.0026628
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027040, upper bound: 0.0026104
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026574, upper bound: 0.0026613
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0023382, upper bound: 0.0023501
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0023349, upper bound: 0.0023520
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026282, upper bound: 0.0027157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026317, upper bound: 0.0027139
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026984, upper bound: 0.0027736
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026970, upper bound: 0.0027736
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026382, upper bound: 0.0025931
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0025441, upper bound: 0.0027035
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026791, upper bound: 0.0027188
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026791, upper bound: 0.0027189
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026147, upper bound: 0.0027354
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026132, upper bound: 0.0027414
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027426, upper bound: 0.0028202
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027417, upper bound: 0.0028209
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027003, upper bound: 0.0027354
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026365, upper bound: 0.0027798
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026893, upper bound: 0.0026262
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026056, upper bound: 0.0027243
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0020003, upper bound: 0.0019924
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0020003, upper bound: 0.0019924
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0025968, upper bound: 0.0026020
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0025430, upper bound: 0.0026559
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0025107, upper bound: 0.0027219
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0024602, upper bound: 0.0027699
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0019280, upper bound: 0.0018399
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0019280, upper bound: 0.0018399
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026353, upper bound: 0.0025932
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0025820, upper bound: 0.0026510
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026974, upper bound: 0.0026389
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026334, upper bound: 0.0026841
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026433, upper bound: 0.0026057
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0025412, upper bound: 0.0026997
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027497, upper bound: 0.0025955
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027485, upper bound: 0.0026008
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0022755, upper bound: 0.0022680
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0022660, upper bound: 0.0022698
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0025735, upper bound: 0.0026204
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0025263, upper bound: 0.0026744
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026297, upper bound: 0.0027163
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026297, upper bound: 0.0027167
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0028203, upper bound: 0.0025396
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027511, upper bound: 0.0026181
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026749, upper bound: 0.0026102
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026794, upper bound: 0.0026029
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0021266, upper bound: 0.0021121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0021266, upper bound: 0.0021121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0022908, upper bound: 0.0023774
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0022903, upper bound: 0.0023845
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0024029, upper bound: 0.0023212
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0023984, upper bound: 0.0023239
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0021189, upper bound: 0.0021654
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0021189, upper bound: 0.0021654
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026727, upper bound: 0.0027228
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026781, upper bound: 0.0027212
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0025941, upper bound: 0.0026681
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0025000, upper bound: 0.0027700
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027479, upper bound: 0.0025752
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026391, upper bound: 0.0026660
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027576, upper bound: 0.0026834
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027086, upper bound: 0.0027589
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0028050, upper bound: 0.0026513
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027644, upper bound: 0.0026778
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0019481, upper bound: 0.0019642
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0019481, upper bound: 0.0019642
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0019963, upper bound: 0.0019985
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0019963, upper bound: 0.0019985
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0027076, upper bound: 0.0028006
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0026668, upper bound: 0.0028479

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0045669, 0.0044577
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0011379, 0.0011107
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0058864, 0.0060305
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0027448, 0.0026792
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0011393, 0.0011672
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0074035, 0.0075848
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0019251, 0.0018791
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0049808, 0.0048618
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0026194, 0.0025568
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0029647, 0.0030373

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026730, upper bound: 0.0024982
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026729, upper bound: 0.0024997
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0045000, 0.0045286
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0011213, 0.0011284
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0059799, 0.0059421
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0027046, 0.0027218
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0011574, 0.0011501
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0075212, 0.0074736
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0018969, 0.0019090
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0049078, 0.0049391
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0025810, 0.0025974
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0030118, 0.0029928

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0018294, upper bound: 0.0018314
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0018294, upper bound: 0.0018314
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0049043, 0.0048522
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012220, 0.0012090
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0064073, 0.0064761
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029476, 0.0029163
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012401, 0.0012534
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0080587, 0.0081452
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020674, 0.0020454
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0053489, 0.0052920
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028129, 0.0027830
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0032271, 0.0032617

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026269, upper bound: 0.0024100
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025171, upper bound: 0.0025096
time: 1.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0048724, 0.0048955
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012141, 0.0012198
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0064644, 0.0064339
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029284, 0.0029423
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012512, 0.0012453
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0081306, 0.0080922
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020539, 0.0020636
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0053140, 0.0053392
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0027946, 0.0028078
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0032558, 0.0032405

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025899, upper bound: 0.0024575
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0024801, upper bound: 0.0025589
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0049509, 0.0048773
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012336, 0.0012153
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0064405, 0.0065376
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029756, 0.0029314
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012465, 0.0012653
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0081004, 0.0082226
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020870, 0.0020560
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0053997, 0.0053194
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028396, 0.0027974
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0032438, 0.0032927

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026935, upper bound: 0.0025318
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026507, upper bound: 0.0025880
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0049386, 0.0048872
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012306, 0.0012178
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0064535, 0.0065214
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029683, 0.0029374
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012491, 0.0012622
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0081168, 0.0082022
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020818, 0.0020601
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0053863, 0.0053302
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028326, 0.0028031
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0032503, 0.0032845

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026692, upper bound: 0.0024642
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025627, upper bound: 0.0025544
time: 1.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0049284, 0.0048659
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012280, 0.0012125
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0064254, 0.0065080
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029621, 0.0029245
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012436, 0.0012596
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0080814, 0.0081853
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020775, 0.0020512
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0053752, 0.0053070
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028267, 0.0027909
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0032362, 0.0032778

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025977, upper bound: 0.0024085
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025031, upper bound: 0.0025095
time: 1.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0048921, 0.0049113
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012190, 0.0012238
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0064853, 0.0064600
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029403, 0.0029518
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012552, 0.0012503
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0081568, 0.0081249
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020622, 0.0020703
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0053355, 0.0053564
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028059, 0.0028169
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0032663, 0.0032536

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026213, upper bound: 0.0026258
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026209, upper bound: 0.0026260
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0048166, 0.0049397
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012002, 0.0012308
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0065228, 0.0063602
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0028949, 0.0029689
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012625, 0.0012310
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0082040, 0.0079995
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020304, 0.0020823
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0052532, 0.0053874
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0027626, 0.0028332
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0032852, 0.0032034

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025243, upper bound: 0.0025074
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0024272, upper bound: 0.0026174
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0048221, 0.0049346
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012015, 0.0012296
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0065161, 0.0063675
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0028982, 0.0029659
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012612, 0.0012324
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0081956, 0.0080087
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020327, 0.0020801
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0052592, 0.0053819
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0027658, 0.0028303
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0032819, 0.0032070

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025920, upper bound: 0.0026320
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025670, upper bound: 0.0026770
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0049308, 0.0050318
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012286, 0.0012538
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0066444, 0.0065111
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029636, 0.0030243
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012860, 0.0012602
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0083569, 0.0081893
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020785, 0.0021211
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0053778, 0.0054879
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028281, 0.0028860
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033465, 0.0032793

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0019087, upper bound: 0.0019469
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0019087, upper bound: 0.0019469
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0049309, 0.0050291
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012287, 0.0012531
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0066408, 0.0065112
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029636, 0.0030226
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012853, 0.0012602
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0083524, 0.0081894
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020786, 0.0021199
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0053779, 0.0054849
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028282, 0.0028845
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033447, 0.0032794

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026569, upper bound: 0.0026800
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026307, upper bound: 0.0027393
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0045028, 0.0045236
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0011220, 0.0011271
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0059733, 0.0059459
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0027063, 0.0027188
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0011561, 0.0011508
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0075128, 0.0074784
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0018981, 0.0019068
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0049110, 0.0049336
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0025826, 0.0025945
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0030085, 0.0029947

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025984, upper bound: 0.0025603
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025960, upper bound: 0.0025603
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0044356, 0.0045898
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0011052, 0.0011437
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0060608, 0.0058571
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0026659, 0.0027586
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0011730, 0.0011336
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0076229, 0.0073668
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0018698, 0.0019348
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0048376, 0.0050058
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0025441, 0.0026325
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0030525, 0.0029500

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0017876, upper bound: 0.0018657
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0017876, upper bound: 0.0018657
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0051855, 0.0051887
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012921, 0.0012929
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0068517, 0.0068474
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031166, 0.0031186
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013261, 0.0013253
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0086176, 0.0086122
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021859, 0.0021872
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0056555, 0.0056590
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029742, 0.0029760
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034509, 0.0034487

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025783, upper bound: 0.0025185
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0024722, upper bound: 0.0026102
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0051860, 0.0051851
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012922, 0.0012920
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0068469, 0.0068481
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031169, 0.0031164
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013252, 0.0013254
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0086116, 0.0086131
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021861, 0.0021857
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0056561, 0.0056551
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029745, 0.0029740
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034485, 0.0034490

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026304, upper bound: 0.0025998
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025752, upper bound: 0.0026707
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0051328, 0.0051590
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012790, 0.0012855
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0068124, 0.0067778
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0030850, 0.0031007
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013185, 0.0013118
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0085682, 0.0085247
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021637, 0.0021747
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0055981, 0.0056266
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029440, 0.0029590
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034311, 0.0034137

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025100, upper bound: 0.0025303
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0024157, upper bound: 0.0026368
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0051245, 0.0051657
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012769, 0.0012872
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0068212, 0.0067669
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0030800, 0.0031047
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013202, 0.0013097
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0085793, 0.0085110
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021602, 0.0021775
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0055890, 0.0056339
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029392, 0.0029628
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034355, 0.0034082

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0018914, upper bound: 0.0019322
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0018914, upper bound: 0.0019322
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0051534, 0.0052204
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012841, 0.0013008
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0068935, 0.0068050
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0030973, 0.0031376
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013342, 0.0013171
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0086701, 0.0085589
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021723, 0.0022006
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0056205, 0.0056936
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029558, 0.0029942
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034719, 0.0034274

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026649, upper bound: 0.0027085
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026204, upper bound: 0.0027476
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0051546, 0.0052182
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012844, 0.0013002
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0068906, 0.0068065
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0030980, 0.0031363
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013337, 0.0013174
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0086665, 0.0085608
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021728, 0.0021997
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0056218, 0.0056912
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029564, 0.0029929
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034704, 0.0034281

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026384, upper bound: 0.0026185
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025413, upper bound: 0.0027143
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0048956, 0.0049285
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012199, 0.0012280
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0065080, 0.0064646
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029424, 0.0029621
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012596, 0.0012512
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0081853, 0.0081307
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020637, 0.0020775
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0053393, 0.0053752
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028079, 0.0028268
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0032778, 0.0032559

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026018, upper bound: 0.0025342
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0024952, upper bound: 0.0026329
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0048451, 0.0049771
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012073, 0.0012402
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0065722, 0.0063979
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029121, 0.0029914
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012720, 0.0012383
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0082661, 0.0080469
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020424, 0.0020980
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0052843, 0.0054282
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0027790, 0.0028547
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033101, 0.0032223

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025341, upper bound: 0.0025784
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0024356, upper bound: 0.0026737
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0048742, 0.0047944
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012145, 0.0011946
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0063310, 0.0064363
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029295, 0.0028816
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012253, 0.0012457
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0079627, 0.0080951
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020546, 0.0020210
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0053160, 0.0052290
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0027956, 0.0027499
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0031886, 0.0032416

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0018530, upper bound: 0.0018091
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0018530, upper bound: 0.0018091
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0048124, 0.0048606
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0011991, 0.0012111
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0064184, 0.0063548
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0028924, 0.0029214
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012423, 0.0012299
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0080727, 0.0079926
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020286, 0.0020489
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0052486, 0.0053012
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0027602, 0.0027879
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0032326, 0.0032006

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025643, upper bound: 0.0026909
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025509, upper bound: 0.0026911
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0044685, 0.0045672
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0011134, 0.0011380
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0060310, 0.0059007
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0026857, 0.0027450
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0011673, 0.0011421
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0075854, 0.0074215
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0018837, 0.0019252
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0048736, 0.0049812
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0025630, 0.0026196
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0030375, 0.0029719

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025022, upper bound: 0.0026211
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0024978, upper bound: 0.0026211
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0046691, 0.0048129
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0011634, 0.0011992
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0063554, 0.0061655
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0028063, 0.0028927
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012301, 0.0011933
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0079934, 0.0077545
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0019682, 0.0020288
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0050923, 0.0052491
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0026780, 0.0027605
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0032009, 0.0031053

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0024696, upper bound: 0.0026823
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0024677, upper bound: 0.0026827
time: 1.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0046290, 0.0048492
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0011534, 0.0012083
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0064032, 0.0061126
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0027822, 0.0029145
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012393, 0.0011831
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0080536, 0.0076881
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0019513, 0.0020441
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0050486, 0.0052887
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0026550, 0.0027813
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0032250, 0.0030786

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0024299, upper bound: 0.0027345
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0024300, upper bound: 0.0027345
time: 1.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0051237, 0.0050682
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012767, 0.0012629
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0066925, 0.0067658
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0030795, 0.0030462
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012953, 0.0013095
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0084175, 0.0085097
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021598, 0.0021364
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0055882, 0.0055276
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029388, 0.0029069
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033707, 0.0034076

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0018255, upper bound: 0.0018016
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0018255, upper bound: 0.0018016
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0050741, 0.0051233
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012643, 0.0012766
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0067652, 0.0067002
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0030497, 0.0030792
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013094, 0.0012968
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0085089, 0.0084271
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021389, 0.0021597
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0055340, 0.0055877
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029103, 0.0029385
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034073, 0.0033746

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025491, upper bound: 0.0026066
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025491, upper bound: 0.0026095
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0054664, 0.0054184
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013621, 0.0013501
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0071550, 0.0072183
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032854, 0.0032566
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013848, 0.0013971
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0089991, 0.0090787
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023043, 0.0022841
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0059618, 0.0059096
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031353, 0.0031078
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036036, 0.0036355

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026628, upper bound: 0.0026033
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026616, upper bound: 0.0026042
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0054101, 0.0054684
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013480, 0.0013626
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0072210, 0.0071439
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032516, 0.0032867
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013976, 0.0013827
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0090821, 0.0089852
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022805, 0.0023051
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0059004, 0.0059641
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031030, 0.0031365
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036369, 0.0035981

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0018555, upper bound: 0.0018875
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0018555, upper bound: 0.0018875
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0052488, 0.0052694
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013079, 0.0013130
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0069582, 0.0069310
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031547, 0.0031671
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013467, 0.0013415
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0087516, 0.0087174
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022126, 0.0022212
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0057246, 0.0057470
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030105, 0.0030223
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035045, 0.0034908

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026083, upper bound: 0.0025732
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026068, upper bound: 0.0025736
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0051816, 0.0053332
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012911, 0.0013289
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0070425, 0.0068422
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031143, 0.0032054
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013631, 0.0013243
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0088576, 0.0086057
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0021842, 0.0022481
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0056512, 0.0058166
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0029719, 0.0030589
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035470, 0.0034461

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0017498, upper bound: 0.0018214
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0017498, upper bound: 0.0018214
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0052954, 0.0051537
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013195, 0.0012842
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0068053, 0.0069925
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031827, 0.0030975
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013172, 0.0013534
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0085593, 0.0087947
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022322, 0.0021724
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0057754, 0.0056208
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030372, 0.0029559
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034275, 0.0035218

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027311, upper bound: 0.0025000
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026659, upper bound: 0.0025774
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0052966, 0.0051532
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013198, 0.0012840
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0068047, 0.0069941
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031834, 0.0030972
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013170, 0.0013537
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0085585, 0.0087967
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022327, 0.0021722
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0057767, 0.0056203
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030379, 0.0029556
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034272, 0.0035226

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026646, upper bound: 0.0025158
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026670, upper bound: 0.0025080
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0052355, 0.0053532
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013045, 0.0013339
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0070689, 0.0069134
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031467, 0.0032174
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013682, 0.0013381
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0088908, 0.0086952
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022069, 0.0022566
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0057100, 0.0058385
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030028, 0.0030704
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035603, 0.0034819

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0017501, upper bound: 0.0018107
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0017501, upper bound: 0.0018107
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0054770, 0.0055240
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013647, 0.0013764
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0072944, 0.0072323
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032918, 0.0033201
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014118, 0.0013998
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0091745, 0.0090963
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023087, 0.0023286
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0059734, 0.0060248
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031414, 0.0031684
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036739, 0.0036426

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0017732, upper bound: 0.0018092
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0017732, upper bound: 0.0018092
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0054795, 0.0055276
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013653, 0.0013773
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0072991, 0.0072356
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032933, 0.0033222
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014127, 0.0014004
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0091804, 0.0091005
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023098, 0.0023301
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0059761, 0.0060286
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031428, 0.0031704
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036762, 0.0036442

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025484, upper bound: 0.0025857
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0024990, upper bound: 0.0026374
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0047724, 0.0046064
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0011892, 0.0011478
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0060828, 0.0063019
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0028684, 0.0027686
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0011773, 0.0012197
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0076505, 0.0079262
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020117, 0.0019418
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0052050, 0.0050240
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0027373, 0.0026421
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0030636, 0.0031740

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027230, upper bound: 0.0024592
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027260, upper bound: 0.0024554
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0047394, 0.0046337
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0011809, 0.0011546
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0061187, 0.0062583
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0028485, 0.0027850
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0011843, 0.0012113
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0076958, 0.0078713
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0019978, 0.0019533
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0051690, 0.0050537
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0027183, 0.0026577
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0030817, 0.0031520

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027162, upper bound: 0.0025587
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026595, upper bound: 0.0025774
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053215, 0.0053186
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013260, 0.0013253
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0070232, 0.0070270
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0031984, 0.0031966
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013593, 0.0013601
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0088333, 0.0088381
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022432, 0.0022420
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0058039, 0.0058007
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030522, 0.0030505
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035372, 0.0035392

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026357, upper bound: 0.0025383
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026043, upper bound: 0.0025718
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053279, 0.0053140
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013276, 0.0013241
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0070171, 0.0070354
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032022, 0.0031939
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013581, 0.0013617
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0088256, 0.0088487
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022459, 0.0022400
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0058108, 0.0057957
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030559, 0.0030479
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035342, 0.0035434

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026592, upper bound: 0.0024992
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025981, upper bound: 0.0025832
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0054635, 0.0055528
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013614, 0.0013836
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0073324, 0.0072145
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032837, 0.0033374
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014192, 0.0013964
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0092223, 0.0090740
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023031, 0.0023407
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0059588, 0.0060561
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031337, 0.0031849
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036930, 0.0036336

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025698, upper bound: 0.0025205
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0024711, upper bound: 0.0026196
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0054688, 0.0055481
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013627, 0.0013824
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0073262, 0.0072215
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032869, 0.0033346
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014180, 0.0013977
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0092145, 0.0090828
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023053, 0.0023387
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0059645, 0.0060510
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031367, 0.0031822
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036899, 0.0036371

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026377, upper bound: 0.0026480
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026111, upper bound: 0.0026823
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0049626, 0.0050812
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012365, 0.0012661
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0067096, 0.0065530
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029827, 0.0030539
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0012986, 0.0012683
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0084389, 0.0082420
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020919, 0.0021419
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0054124, 0.0055417
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028463, 0.0029143
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0033793, 0.0033005

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0025758, upper bound: 0.0025890
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025013, upper bound: 0.0026494
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0048969, 0.0051461
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0012202, 0.0012823
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0067953, 0.0064663
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0029432, 0.0030929
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013152, 0.0012515
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0085467, 0.0081329
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0020642, 0.0021692
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0053407, 0.0056125
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0028086, 0.0029516
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0034225, 0.0032568

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0024817, upper bound: 0.0026883
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0024093, upper bound: 0.0027511
time: 1.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0054506, 0.0053364
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013582, 0.0013297
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0070467, 0.0071975
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032760, 0.0032074
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013639, 0.0013931
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0088629, 0.0090526
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022976, 0.0022495
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0059447, 0.0058201
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031263, 0.0030608
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035491, 0.0036250

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0027298, upper bound: 0.0024824
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0026679, upper bound: 0.0025562
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0053774, 0.0053986
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013399, 0.0013452
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0071287, 0.0071008
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0032320, 0.0032447
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0013798, 0.0013743
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0089661, 0.0089310
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0022668, 0.0022757
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0058648, 0.0058879
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0030843, 0.0030964
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0035904, 0.0035763

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0026210, upper bound: 0.0025755
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0025543, upper bound: 0.0026474
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9889394, 0.9967324, 0.9889394, 0.9967324, -0.0055646, 0.0055043
1: -0.0040199, -0.0020781, -0.0040199, -0.0020781, -0.0013865, 0.0013715
2: 0.0009591, 0.0112496, 0.0009591, 0.0112496, -0.0072684, 0.0073480
3: -0.0063934, -0.0017097, -0.0063934, -0.0017097, -0.0033445, 0.0033083
4: 0.0007135, 0.0027052, 0.0007135, 0.0027052, -0.0014068, 0.0014222
5: 0.0001657, 0.0131085, 0.0001657, 0.0131085, -0.0091418, 0.0092418
6: -0.0017862, 0.0014988, -0.0017862, 0.0014988, -0.0023457, 0.0023203
7: -0.0077592, 0.0007401, -0.0077592, 0.0007401, -0.0060690, 0.0060033
8: -0.0036446, 0.0008251, -0.0036446, 0.0008251, -0.0031916, 0.0031571
9: -0.0028206, 0.0023623, -0.0028206, 0.0023623, -0.0036608, 0.0037008

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.09 + 598.11 = 601.20 seconds
