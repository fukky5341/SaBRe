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
execution time: IAR + RelationalAnalysis = 24.44 + 33.40 = 57.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.2973304, upper bound: 0.2973307

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5734

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2922977, upper bound: 0.2973261
time: 5.26 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973263, upper bound: 0.2922990
time: 3.66 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.94 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.94
Output dim: 1, lower bound: -0.2922977, upper bound: 0.2973261
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.94
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

Time for backsubstitution: 23.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 820

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2919466, upper bound: 0.2961088
time: 5.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2919466, upper bound: 0.2919740
time: 5.12 seconds

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

Time for backsubstitution: 23.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6220

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973170, upper bound: 0.2922995
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973262, upper bound: 0.2922887
time: 3.54 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.48 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.48
Output dim: 1, lower bound: -0.2919466, upper bound: 0.2961088
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 30.48
Output dim: 1, lower bound: -0.2919466, upper bound: 0.2919740
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.48
Output dim: 1, lower bound: -0.2973170, upper bound: 0.2922995
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.48
Output dim: 1, lower bound: -0.2973262, upper bound: 0.2922887

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7252338, 0.7202425
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5160316, 0.5210084
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5167193, 0.5127805
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5721784, 0.5632199
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.5990958, 0.6099880
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5595479, 0.5597086
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6152170, 0.6175776
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4081941, 0.4045283
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4074842, 0.4033886
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4622703, 0.4546825

Time for backsubstitution: 23.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2919381, upper bound: 0.2961080
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2919466, upper bound: 0.2961004
time: 6.56 seconds

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

Time for backsubstitution: 23.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5829

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2962276, upper bound: 0.2922928
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2973090, upper bound: 0.2914880
time: 4.77 seconds

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

Time for backsubstitution: 23.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 820

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2919738, upper bound: 0.2919363
time: 5.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2961088, upper bound: 0.2919382
time: 5.12 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 34.05 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.05
Output dim: 1, lower bound: -0.2919381, upper bound: 0.2961080
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.05
Output dim: 1, lower bound: -0.2919466, upper bound: 0.2961004
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.05
Output dim: 1, lower bound: -0.2962276, upper bound: 0.2922928
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.05
Output dim: 1, lower bound: -0.2973090, upper bound: 0.2914880
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 34.05
Output dim: 1, lower bound: -0.2919738, upper bound: 0.2919363
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.05
Output dim: 1, lower bound: -0.2961088, upper bound: 0.2919382

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7240186, 0.7176342
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5143970, 0.5202489
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5145767, 0.5081728
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5661542, 0.5604193
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.5971875, 0.6091008
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5587065, 0.5593181
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6097720, 0.6150447
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4048710, 0.4029840
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4046296, 0.3972565
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4592835, 0.4482719

Time for backsubstitution: 23.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5829

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2909284, upper bound: 0.2961014
time: 6.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2919300, upper bound: 0.2953351
time: 3.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7226257, 0.7190270
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5152720, 0.5193739
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5121114, 0.5106381
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5693779, 0.5571955
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.5982087, 0.6080797
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5591574, 0.5588672
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6126842, 0.6121325
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4066500, 0.4012051
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4013520, 0.4005340
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4558598, 0.4516957

Time for backsubstitution: 23.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5829

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2911728, upper bound: 0.2960914
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2919399, upper bound: 0.2950926
time: 3.57 seconds

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

Time for backsubstitution: 23.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 820

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2909556, upper bound: 0.2919399
time: 3.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2950907, upper bound: 0.2919417
time: 3.83 seconds

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

Time for backsubstitution: 23.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 820

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 820

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2919572, upper bound: 0.2911699
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2960923, upper bound: 0.2911717
time: 3.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7176342, 0.7238474
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5202490, 0.5143684
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5081728, 0.5144417
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5604193, 0.5660980
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.6088147, 0.5971875
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5593181, 0.5586734
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6150446, 0.6097031
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4029405, 0.4048710
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.3972565, 0.4045611
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4482719, 0.4591759

Time for backsubstitution: 23.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5829

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2953353, upper bound: 0.2919288
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2961022, upper bound: 0.2909283
time: 3.46 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.48 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.48
Output dim: 1, lower bound: -0.2909284, upper bound: 0.2961014
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.48
Output dim: 1, lower bound: -0.2919300, upper bound: 0.2953351
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.48
Output dim: 1, lower bound: -0.2911728, upper bound: 0.2960914
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.48
Output dim: 1, lower bound: -0.2919399, upper bound: 0.2950926
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.48
Output dim: 1, lower bound: -0.2909556, upper bound: 0.2919399
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.48
Output dim: 1, lower bound: -0.2950907, upper bound: 0.2919417
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.48
Output dim: 1, lower bound: -0.2919572, upper bound: 0.2911699
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.48
Output dim: 1, lower bound: -0.2960923, upper bound: 0.2911717
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.48
Output dim: 1, lower bound: -0.2953353, upper bound: 0.2919288
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.48
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

Time for backsubstitution: 23.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2489
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 740
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1781
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1964
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 767
type: DSZ, layer: 3, pos: 1164
type: DSZ, layer: 3, pos: 1997
type: DSZ, layer: 3, pos: 1159
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2489

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2894603, upper bound: 0.2960760
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2909081, upper bound: 0.2946254
time: 3.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 23.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 740
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1159
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 767
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1964
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1781
type: DSZ, layer: 3, pos: 1997
type: DSZ, layer: 3, pos: 1164
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 2489
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 2124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1397

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2914696, upper bound: 0.2946422
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2912620, upper bound: 0.2948498
time: 3.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 23.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1164
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 767
type: DSZ, layer: 3, pos: 1781
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1964
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 740
type: DSZ, layer: 3, pos: 1997
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1159
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2489
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1397

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1164

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2880339, upper bound: 0.2932714
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2880534, upper bound: 0.2931628
time: 3.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 23.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1159
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1164
type: DSZ, layer: 3, pos: 1997
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2489
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 767
type: DSZ, layer: 3, pos: 1781
type: DSZ, layer: 3, pos: 1964
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 740
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1794

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2124

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2864845, upper bound: 0.2948690
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2916946, upper bound: 0.2881980
time: 3.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7193444, 0.7222903
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5183789, 0.5170429
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5127964, 0.5107780
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5560849, 0.5713431
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.6067467, 0.6000996
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5582051, 0.5603161
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6119287, 0.6129825
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4010645, 0.4068249
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.4022542, 0.4003313
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4533681, 0.4548285

Time for backsubstitution: 23.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1997
type: DSZ, layer: 3, pos: 1781
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 2489
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1164
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1159
type: DSZ, layer: 3, pos: 767
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 740
type: DSZ, layer: 3, pos: 1964
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 227

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1997

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2931793, upper bound: 0.2898494
time: 4.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2928848, upper bound: 0.2898492
time: 4.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7188632, 0.7224543
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5193740, 0.5142483
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5094397, 0.5119764
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5571954, 0.5682110
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.6077933, 0.5971625
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5588672, 0.5584621
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6121325, 0.6124113
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4011615, 0.4065531
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.3995817, 0.4012836
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4507722, 0.4557521

Time for backsubstitution: 23.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 1964
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1164
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1997
type: DSZ, layer: 3, pos: 740
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1781
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2489
type: DSZ, layer: 3, pos: 767
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1159
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2536

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2124

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2891943, upper bound: 0.2909316
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2958691, upper bound: 0.2857046
time: 3.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.6370115, -7.4326506, -8.6370115, -7.4326506, -0.7179520, 0.7236834
1: 2.8181701, 3.7671344, 2.8181701, 3.7671344, -0.5192536, 0.5161692
2: -6.2361569, -5.2770119, -6.2361569, -5.2770119, -0.5103338, 0.5132434
3: -11.2083759, -10.0195055, -11.2083759, -10.0195055, -0.5593085, 0.5681183
4: -4.0244241, -3.0589385, -4.0244241, -3.0589385, -0.6077681, 0.5990798
5: -12.1397896, -11.0226450, -12.1397896, -11.0226450, -0.5586560, 0.5598662
6: -9.7727098, -8.5192518, -9.7727098, -8.5192518, -0.6148407, 0.6100714
7: -3.9430108, -2.9613657, -3.9430108, -2.9613657, -0.4028435, 0.4050459
8: -3.1051531, -2.1637201, -3.1051531, -2.1637201, -0.3989778, 0.4036087
9: -11.8276062, -10.8258801, -11.8276062, -10.8258801, -0.4499460, 0.4582522

Time for backsubstitution: 23.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2489
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 767
type: DSZ, layer: 3, pos: 1159
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1964
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1781
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 1997
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1794
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1164
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 740

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2489

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2938634, upper bound: 0.2919042
time: 5.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2953132, upper bound: 0.2904484
time: 6.26 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 35.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 35.50
Output dim: 1, lower bound: -0.2894603, upper bound: 0.2960760
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 35.50
Output dim: 1, lower bound: -0.2909081, upper bound: 0.2946254
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 35.50
Output dim: 1, lower bound: -0.2914696, upper bound: 0.2946422
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 35.50
Output dim: 1, lower bound: -0.2912620, upper bound: 0.2948498
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 35.50
Output dim: 1, lower bound: -0.2880339, upper bound: 0.2932714
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 35.50
Output dim: 1, lower bound: -0.2880534, upper bound: 0.2931628
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 35.50
Output dim: 1, lower bound: -0.2864845, upper bound: 0.2948690
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 35.50
Output dim: 1, lower bound: -0.2916946, upper bound: 0.2881980
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 35.50
Output dim: 1, lower bound: -0.2931793, upper bound: 0.2898494
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 35.50
Output dim: 1, lower bound: -0.2928848, upper bound: 0.2898492
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 35.50
Output dim: 1, lower bound: -0.2891943, upper bound: 0.2909316
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 35.50
Output dim: 1, lower bound: -0.2958691, upper bound: 0.2857046
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 35.50
Output dim: 1, lower bound: -0.2938634, upper bound: 0.2919042
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 35.50
Output dim: 1, lower bound: -0.2953132, upper bound: 0.2904484
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 1, lower bound: -0.2961022, upper bound: 0.2909283

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.84 + 547.74 = 605.58 seconds
