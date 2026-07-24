## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00056538


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005468, 0.0005468)
1: (0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010591, 0.0010591)
2: (-0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0085423, 0.0085423)
3: (-0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007630, 0.0007630)
4: (0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0037017, 0.0037017)
5: (-0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005526, 0.0005526)
6: (0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0010135, 0.0010135)
7: (0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0067008, 0.0067008)
8: (0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020993, 0.0020993)
9: (-0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0041899, 0.0041899)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 2.32 = 3.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0006282, upper bound: 0.0006282

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006135, upper bound: 0.0005935
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005934, upper bound: 0.0006135
time: 1.32 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.75 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.75
Output dim: 6, lower bound: -0.0006135, upper bound: 0.0005935
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.75
Output dim: 6, lower bound: -0.0005934, upper bound: 0.0006135

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005253, 0.0005187
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010175, 0.0010046
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0081031, 0.0082073
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007330, 0.0007237
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0035565, 0.0035114
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005242, 0.0005309
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009737, 0.0009614
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0064379, 0.0063562
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020170, 0.0019914
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0039745, 0.0040256

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006013, upper bound: 0.0005660
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005661, upper bound: 0.0005795
time: 2.03 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005187, 0.0005253
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010046, 0.0010175
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0082073, 0.0081031
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007237, 0.0007330
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0035114, 0.0035565
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005309, 0.0005242
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009614, 0.0009737
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0063562, 0.0064379
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0019914, 0.0020170
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0040256, 0.0039745

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005794, upper bound: 0.0005767
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005661, upper bound: 0.0006013
time: 1.33 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.60 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 6, lower bound: -0.0006013, upper bound: 0.0005660
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 6, lower bound: -0.0005661, upper bound: 0.0005795
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 6, lower bound: -0.0005794, upper bound: 0.0005767
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 6, lower bound: -0.0005661, upper bound: 0.0006013

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005070, 0.0004923
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009819, 0.0009535
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0076911, 0.0079202
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007074, 0.0006869
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0034321, 0.0033329
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0004975, 0.0005123
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009397, 0.0009125
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0062127, 0.0060330
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0019464, 0.0018901
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0037724, 0.0038848

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005532, upper bound: 0.0005534
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005857, upper bound: 0.0005531
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0004990, 0.0005000
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009664, 0.0009685
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0078122, 0.0077953
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006962, 0.0006977
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0033780, 0.0033853
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005054, 0.0005043
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009249, 0.0009269
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0061147, 0.0061280
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0019157, 0.0019199
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0038318, 0.0038235

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005532, upper bound: 0.0005661
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005532, upper bound: 0.0005661
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005000, 0.0004990
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009685, 0.0009664
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0077953, 0.0078122
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006977, 0.0006962
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0033853, 0.0033780
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005043, 0.0005054
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009269, 0.0009249
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0061280, 0.0061147
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0019199, 0.0019157
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0038235, 0.0038318

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005532, upper bound: 0.0005640
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005661, upper bound: 0.0005638
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0004923, 0.0005070
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009535, 0.0009819
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0079202, 0.0076911
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006869, 0.0007074
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0033329, 0.0034321
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005123, 0.0004975
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009125, 0.0009397
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0060330, 0.0062127
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0018901, 0.0019464
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0038848, 0.0037724

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005532, upper bound: 0.0005856
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005532, upper bound: 0.0005869
time: 2.12 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.80 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.80
Output dim: 6, lower bound: -0.0005532, upper bound: 0.0005534
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.80
Output dim: 6, lower bound: -0.0005857, upper bound: 0.0005531
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.80
Output dim: 6, lower bound: -0.0005532, upper bound: 0.0005661
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.80
Output dim: 6, lower bound: -0.0005532, upper bound: 0.0005661
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.80
Output dim: 6, lower bound: -0.0005532, upper bound: 0.0005640
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.80
Output dim: 6, lower bound: -0.0005661, upper bound: 0.0005638
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.80
Output dim: 6, lower bound: -0.0005532, upper bound: 0.0005856
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.80
Output dim: 6, lower bound: -0.0005532, upper bound: 0.0005869

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0004963, 0.0004830
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009613, 0.0009355
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0075460, 0.0077541
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006926, 0.0006740
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0033601, 0.0032700
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0004881, 0.0005016
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009200, 0.0008953
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0060824, 0.0059192
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0019056, 0.0018545
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0037012, 0.0038033

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005782, upper bound: 0.0005259
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005259, upper bound: 0.0005457
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0004897, 0.0004897
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009485, 0.0009484
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0076499, 0.0076502
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006833, 0.0006833
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0033151, 0.0033150
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0004949, 0.0004949
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009076, 0.0009076
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0060009, 0.0060007
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0018800, 0.0018800
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0037522, 0.0037523

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005563, upper bound: 0.0005423
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005259, upper bound: 0.0005586
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0004884, 0.0004908
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009460, 0.0009506
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0076671, 0.0076306
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006815, 0.0006848
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0033066, 0.0033224
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0004960, 0.0004936
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009053, 0.0009096
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0059856, 0.0060142
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0018752, 0.0018842
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0037606, 0.0037427

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005564, upper bound: 0.0005422
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005259, upper bound: 0.0005586
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0004897, 0.0004897
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009484, 0.0009485
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0076502, 0.0076499
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006833, 0.0006833
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0033150, 0.0033151
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0004949, 0.0004949
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009076, 0.0009076
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0060007, 0.0060009
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0018800, 0.0018800
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0037523, 0.0037522

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005586, upper bound: 0.0005406
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005422, upper bound: 0.0005562
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0004830, 0.0004963
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009355, 0.0009613
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0077541, 0.0075460
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006740, 0.0006926
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0032700, 0.0033601
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005016, 0.0004881
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0008953, 0.0009200
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0059192, 0.0060824
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0018545, 0.0019056
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0038033, 0.0037012

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005458, upper bound: 0.0005641
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005259, upper bound: 0.0005782
time: 1.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0004821, 0.0004977
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009338, 0.0009639
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0077751, 0.0075323
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006727, 0.0006944
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0032641, 0.0033692
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005030, 0.0004873
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0008937, 0.0009225
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0059085, 0.0060989
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0018511, 0.0019107
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0038136, 0.0036945

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005460, upper bound: 0.0005644
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005259, upper bound: 0.0005793
time: 1.44 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.16 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 6, lower bound: -0.0005782, upper bound: 0.0005259
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.16
Output dim: 6, lower bound: -0.0005259, upper bound: 0.0005457
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.16
Output dim: 6, lower bound: -0.0005563, upper bound: 0.0005423
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.16
Output dim: 6, lower bound: -0.0005259, upper bound: 0.0005586
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.16
Output dim: 6, lower bound: -0.0005564, upper bound: 0.0005422
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.16
Output dim: 6, lower bound: -0.0005259, upper bound: 0.0005586
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.16
Output dim: 6, lower bound: -0.0005586, upper bound: 0.0005406
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.16
Output dim: 6, lower bound: -0.0005422, upper bound: 0.0005562
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.16
Output dim: 6, lower bound: -0.0005458, upper bound: 0.0005641
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 6, lower bound: -0.0005259, upper bound: 0.0005782
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.16
Output dim: 6, lower bound: -0.0005460, upper bound: 0.0005644
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 6, lower bound: -0.0005259, upper bound: 0.0005793

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0004923, 0.0004766
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009535, 0.0009232
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0074464, 0.0076911
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006869, 0.0006651
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0033329, 0.0032268
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0004817, 0.0004975
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009125, 0.0008835
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0060331, 0.0058411
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0018901, 0.0018300
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0036524, 0.0037724

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005688, upper bound: 0.0005141
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005563, upper bound: 0.0005170
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0004766, 0.0004923
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009232, 0.0009535
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0076911, 0.0074464
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006651, 0.0006869
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0032268, 0.0033329
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0004975, 0.0004817
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0008835, 0.0009125
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0058411, 0.0060331
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0018300, 0.0018901
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0037724, 0.0036524

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005141, upper bound: 0.0005563
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005140, upper bound: 0.0005688
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0004758, 0.0004937
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009215, 0.0009562
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0077128, 0.0074327
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006639, 0.0006889
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0032209, 0.0033422
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0004989, 0.0004808
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0008818, 0.0009151
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0058304, 0.0060500
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0018266, 0.0018954
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0037830, 0.0036457

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005141, upper bound: 0.0005563
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005141, upper bound: 0.0005699
time: 2.15 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.89 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 6, lower bound: -0.0005688, upper bound: 0.0005141
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.89
Output dim: 6, lower bound: -0.0005563, upper bound: 0.0005170
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.89
Output dim: 6, lower bound: -0.0005141, upper bound: 0.0005563
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 6, lower bound: -0.0005140, upper bound: 0.0005688
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.89
Output dim: 6, lower bound: -0.0005141, upper bound: 0.0005563
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 6, lower bound: -0.0005141, upper bound: 0.0005699

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0004918, 0.0004759
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009526, 0.0009218
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0074355, 0.0076834
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006862, 0.0006641
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0033295, 0.0032221
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0004810, 0.0004970
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009116, 0.0008822
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0060270, 0.0058326
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0018882, 0.0018273
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0036470, 0.0037686

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005015, upper bound: 0.0004601
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005015, upper bound: 0.0004601
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0004759, 0.0004918
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009218, 0.0009526
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0076834, 0.0074355
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006641, 0.0006862
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0032221, 0.0033295
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0004970, 0.0004810
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0008822, 0.0009116
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0058326, 0.0060270
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0018273, 0.0018882
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0037686, 0.0036470

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004600, upper bound: 0.0005016
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004600, upper bound: 0.0005016
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0004751, 0.0004932
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009202, 0.0009553
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0077050, 0.0074219
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006629, 0.0006882
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0032162, 0.0033389
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0004984, 0.0004801
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0008806, 0.0009141
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0058219, 0.0060440
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0018240, 0.0018935
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0037792, 0.0036404

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004605, upper bound: 0.0005016
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004605, upper bound: 0.0005016
time: 1.31 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.92 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 6, lower bound: -0.0005015, upper bound: 0.0004601
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 6, lower bound: -0.0005015, upper bound: 0.0004601
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 6, lower bound: -0.0004600, upper bound: 0.0005016
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 6, lower bound: -0.0004600, upper bound: 0.0005016
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 6, lower bound: -0.0004605, upper bound: 0.0005016
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 6, lower bound: -0.0004605, upper bound: 0.0005016

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.63 + 79.68 = 83.32 seconds
