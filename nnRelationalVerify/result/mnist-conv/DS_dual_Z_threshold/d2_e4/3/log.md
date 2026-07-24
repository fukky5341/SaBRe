## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.294357096


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7342215, 0.7342215)
1: (2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5336686, 0.5336686)
2: (-6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5237415, 0.5237415)
3: (-11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5927124, 0.5927125)
4: (-4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.6310592, 0.6310592)
5: (-12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5609331, 0.5609331)
6: (-9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6248200, 0.6248200)
7: (-3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4181032, 0.4181032)
8: (-3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4160961, 0.4160961)
9: (-11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4786108, 0.4786108)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.61 + 33.71 = 56.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.2973304, upper bound: 0.2973307

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 5734

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922977, upper bound: 0.2973261
time: 5.47 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973263, upper bound: 0.2922990
time: 3.76 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.49 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.49
Output dim: 1, lower bound: -0.2922977, upper bound: 0.2973261
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.49
Output dim: 1, lower bound: -0.2973263, upper bound: 0.2922990

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7250624, 0.7212579
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5160030, 0.5211830
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5165842, 0.5136122
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5721223, 0.5635654
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.6008272, 0.6097016
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5595148, 0.5599315
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6151478, 0.6179850
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4084802, 0.4044847
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4074159, 0.4038156
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4621626, 0.4553556

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 6220

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922892, upper bound: 0.2973251
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922910, upper bound: 0.2973176
time: 3.69 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7212577, 0.7250624
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5211831, 0.5160029
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5136123, 0.5165842
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5635655, 0.5721222
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.6097014, 0.6008272
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5599315, 0.5595148
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6179850, 0.6151478
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4044847, 0.4084802
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4038157, 0.4074159
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4553555, 0.4621626

Time for backsubstitution: 21.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 6220

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973170, upper bound: 0.2922995
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973262, upper bound: 0.2922887
time: 3.73 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.79 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.79
Output dim: 1, lower bound: -0.2922892, upper bound: 0.2973251
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.79
Output dim: 1, lower bound: -0.2922910, upper bound: 0.2973176
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.79
Output dim: 1, lower bound: -0.2973170, upper bound: 0.2922995
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.79
Output dim: 1, lower bound: -0.2973262, upper bound: 0.2922887

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7238474, 0.7186499
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5143684, 0.5204237
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5144417, 0.5090044
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5660980, 0.5607648
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.5989192, 0.6088147
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5586734, 0.5595407
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6097031, 0.6154522
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4051569, 0.4029404
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4045612, 0.3976834
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4591759, 0.4489452

Time for backsubstitution: 22.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 5829

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2912225, upper bound: 0.2973188
time: 4.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922811, upper bound: 0.2964893
time: 3.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7224541, 0.7200427
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5152434, 0.5195487
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5119765, 0.5114697
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5693216, 0.5575410
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.5999403, 0.6077933
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5591240, 0.5590901
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6126151, 0.6125400
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4069359, 0.4011614
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4012836, 0.4009609
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4557523, 0.4523690

Time for backsubstitution: 22.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 5829

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914886, upper bound: 0.2973096
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922910, upper bound: 0.2962291
time: 3.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7200427, 0.7224543
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5195485, 0.5152434
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5114698, 0.5119764
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5575409, 0.5693216
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.6077933, 0.5999403
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5590901, 0.5591240
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6125400, 0.6126151
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4011615, 0.4069358
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4009609, 0.4012836
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4523689, 0.4557521

Time for backsubstitution: 22.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 5829

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2962276, upper bound: 0.2922928
time: 4.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973090, upper bound: 0.2914880
time: 5.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7186499, 0.7238474
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5204235, 0.5143684
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5090044, 0.5144417
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5607648, 0.5660980
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.6088147, 0.5989192
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5595407, 0.5586734
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6154521, 0.6097031
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4029405, 0.4051569
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.3976835, 0.4045611
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4489452, 0.4591759

Time for backsubstitution: 22.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 5829

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964899, upper bound: 0.2922813
time: 4.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973196, upper bound: 0.2912220
time: 3.72 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.53 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.53
Output dim: 1, lower bound: -0.2912225, upper bound: 0.2973188
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.53
Output dim: 1, lower bound: -0.2922811, upper bound: 0.2964893
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.53
Output dim: 1, lower bound: -0.2914886, upper bound: 0.2973096
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.53
Output dim: 1, lower bound: -0.2922910, upper bound: 0.2962291
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.53
Output dim: 1, lower bound: -0.2962276, upper bound: 0.2922928
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.53
Output dim: 1, lower bound: -0.2973090, upper bound: 0.2914880
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.53
Output dim: 1, lower bound: -0.2964899, upper bound: 0.2922813
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.53
Output dim: 1, lower bound: -0.2973196, upper bound: 0.2912220

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7241647, 0.7184858
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5133733, 0.5222230
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5166001, 0.5078061
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5649874, 0.5627861
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.5978725, 0.6107051
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5580113, 0.5607326
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6094992, 0.6158196
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4050600, 0.4031153
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4062814, 0.3967311
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4608483, 0.4480215

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 820

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2909284, upper bound: 0.2961014
time: 6.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2909284, upper bound: 0.2919690
time: 3.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7236836, 0.7186499
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5143684, 0.5194284
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5132434, 0.5090044
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5660980, 0.5596541
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.5989192, 0.6077681
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5586734, 0.5588789
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6097031, 0.6152482
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4051569, 0.4028435
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4036088, 0.3976834
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4582522, 0.4489452

Time for backsubstitution: 22.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 820

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2919300, upper bound: 0.2953351
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2919282, upper bound: 0.2912020
time: 4.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7227719, 0.7198787
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5142483, 0.5213493
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5141375, 0.5102714
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5682111, 0.5595615
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.5988936, 0.6096854
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5584621, 0.5602829
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6124113, 0.6129084
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4068390, 0.4013362
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4030048, 0.4000086
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4574261, 0.4514452

Time for backsubstitution: 22.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 820

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2911728, upper bound: 0.2960914
time: 4.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2909284, upper bound: 0.2919564
time: 4.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7222903, 0.7200427
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5152434, 0.5185534
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5107780, 0.5114697
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5693216, 0.5564303
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.5999403, 0.6067467
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5591240, 0.5584280
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6126151, 0.6123362
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4069359, 0.4010645
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4003313, 0.4009609
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4548285, 0.4523690

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 820

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2919399, upper bound: 0.2950926
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2919381, upper bound: 0.2909575
time: 3.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7203600, 0.7222903
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5185534, 0.5170429
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5136282, 0.5107780
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5564303, 0.5713431
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.6067467, 0.6018310
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5584280, 0.5603161
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6123362, 0.6129825
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4010645, 0.4071107
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4026811, 0.4003313
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4540412, 0.4548285

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 820

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2909556, upper bound: 0.2919399
time: 4.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2950907, upper bound: 0.2919417
time: 3.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7198789, 0.7224543
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5195485, 0.5142483
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5102713, 0.5119764
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5575409, 0.5682110
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.6077933, 0.5988939
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5590901, 0.5584621
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6125400, 0.6124113
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4011615, 0.4068389
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4000086, 0.4012836
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4514453, 0.4557521

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 820

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2919572, upper bound: 0.2911699
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2960923, upper bound: 0.2911717
time: 3.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7189677, 0.7236834
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5194284, 0.5161692
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5111656, 0.5132434
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5596540, 0.5681183
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.6077681, 0.6008112
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5588789, 0.5598662
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6152482, 0.6100714
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4028435, 0.4053317
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.3994046, 0.4036087
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4506192, 0.4582522

Time for backsubstitution: 22.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 820

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2912001, upper bound: 0.2919273
time: 4.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2953353, upper bound: 0.2919288
time: 4.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7184861, 0.7238474
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5204235, 0.5133733
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5078061, 0.5144417
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5607648, 0.5649873
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.6088147, 0.5978725
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5595407, 0.5580113
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6154521, 0.6094991
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4029405, 0.4050599
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.3967311, 0.4045611
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4480214, 0.4591759

Time for backsubstitution: 22.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 820

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2919671, upper bound: 0.2909284
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2961022, upper bound: 0.2909283
time: 3.65 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.73 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2909284, upper bound: 0.2961014
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2909284, upper bound: 0.2919690
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2919300, upper bound: 0.2953351
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2919282, upper bound: 0.2912020
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2911728, upper bound: 0.2960914
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2909284, upper bound: 0.2919564
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2919399, upper bound: 0.2950926
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2919381, upper bound: 0.2909575
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2909556, upper bound: 0.2919399
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2950907, upper bound: 0.2919417
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2919572, upper bound: 0.2911699
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2960923, upper bound: 0.2911717
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2912001, upper bound: 0.2919273
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2953353, upper bound: 0.2919288
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2919671, upper bound: 0.2909284
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 1, lower bound: -0.2961022, upper bound: 0.2909283

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7243359, 0.7174702
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5134019, 0.5220484
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5167351, 0.5069743
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5650434, 0.5624406
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.5961411, 0.6109917
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5580447, 0.5605099
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6095681, 0.6154121
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4047741, 0.4031590
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4063498, 0.3963042
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4609560, 0.4473484

Time for backsubstitution: 22.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 767
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1997
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2489
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 1781
type: DSZ, layer: 3, pos: 1159
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1964
type: DSZ, layer: 3, pos: 740
type: DSZ, layer: 3, pos: 1164

Time for candidate selection: 0.51 seconds

### Candidate
type: DSZ, layer: 3, pos: 1403

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2744281, upper bound: 0.2923407
time: 6.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2876111, upper bound: 0.2798185
time: 3.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7238543, 0.7176342
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5143970, 0.5192537
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5133784, 0.5081728
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5661542, 0.5593086
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.5971875, 0.6080546
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5587065, 0.5586560
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6097720, 0.6148407
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4048710, 0.4028872
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4036772, 0.3972565
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4583601, 0.4482719

Time for backsubstitution: 22.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 767
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1997
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2489
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 1781
type: DSZ, layer: 3, pos: 1159
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1964
type: DSZ, layer: 3, pos: 740
type: DSZ, layer: 3, pos: 1164

Time for candidate selection: 0.42 seconds

### Candidate
type: DSZ, layer: 3, pos: 1403

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2756482, upper bound: 0.2917934
time: 5.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2881661, upper bound: 0.2785958
time: 3.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7229431, 0.7188630
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5142769, 0.5211747
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5142725, 0.5094397
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5682673, 0.5592160
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.5971625, 0.6099720
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5584953, 0.5600600
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6124802, 0.6125009
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4065532, 0.4013801
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4030732, 0.3995817
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4575340, 0.4507720

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 767
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1997
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2489
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 1781
type: DSZ, layer: 3, pos: 1159
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1964
type: DSZ, layer: 3, pos: 740
type: DSZ, layer: 3, pos: 1164

Time for candidate selection: 0.41 seconds

### Candidate
type: DSZ, layer: 3, pos: 1403

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2744340, upper bound: 0.2923279
time: 4.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2876313, upper bound: 0.2798104
time: 3.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7224615, 0.7190270
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5152720, 0.5183787
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5109131, 0.5106381
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5693779, 0.5560849
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.5982087, 0.6070333
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5591574, 0.5582051
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6126842, 0.6119287
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4066500, 0.4011083
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4003997, 0.4005340
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4549361, 0.4516957

Time for backsubstitution: 22.31 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.33 + 554.99 = 611.32 seconds
