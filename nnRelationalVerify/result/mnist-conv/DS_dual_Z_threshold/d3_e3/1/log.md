## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.719649471


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4588056, 1.4588056)
1: (-10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6418862, 1.6418862)
2: (-4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3488832, 1.3488827)
3: (-5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7874470, 1.7874465)
4: (-13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5689108, 1.5689108)
5: (-3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9303412, 0.9303412)
6: (-10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3711376, 1.3711374)
7: (-9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0479746, 2.0479746)
8: (9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5339589, 1.5339584)
9: (-7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8485889, 1.8485889)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.75 + 37.76 = 61.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.7232643, upper bound: 0.7232643

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4556
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 5832
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 4556

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7216273, upper bound: 0.7232577
time: 6.02 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232573, upper bound: 0.7216289
time: 5.02 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 11.31 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 11.31
Output dim: 8, lower bound: -0.7216273, upper bound: 0.7232577
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 11.31
Output dim: 8, lower bound: -0.7232573, upper bound: 0.7216289

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4570236, 1.4583867
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6453323, 1.6432238
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3306108, 1.3349905
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7729244, 1.7700305
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5524387, 1.5461175
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9123406, 0.9087331
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3559046, 1.3525348
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0457077, 2.0475640
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5275140, 1.5285888
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8376226, 1.8394465

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 5832
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 5830

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7215173, upper bound: 0.7232553
time: 5.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7216251, upper bound: 0.7231480
time: 6.05 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4583864, 1.4570236
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6432238, 1.6453323
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3349905, 1.3306108
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7700310, 1.7729244
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5461178, 1.5524387
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9087331, 0.9123406
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3525348, 1.3559046
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0475636, 2.0457072
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5285888, 1.5275140
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8394470, 1.8376226

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 5832
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 5830

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7231472, upper bound: 0.7216256
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232550, upper bound: 0.7215174
time: 6.32 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 34.20 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 34.20
Output dim: 8, lower bound: -0.7215173, upper bound: 0.7232553
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 34.20
Output dim: 8, lower bound: -0.7216251, upper bound: 0.7231480
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 34.20
Output dim: 8, lower bound: -0.7231472, upper bound: 0.7216256
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 34.20
Output dim: 8, lower bound: -0.7232550, upper bound: 0.7215174

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4576674, 1.4581654
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6460600, 1.6429811
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3306737, 1.3349686
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7724991, 1.7712646
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5520368, 1.5472851
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9120483, 0.9095819
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3553119, 1.3542690
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0475039, 2.0469489
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5276589, 1.5285425
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8387084, 1.8390718

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5832
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 5832

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7196434, upper bound: 0.7232548
time: 7.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7215156, upper bound: 0.7213823
time: 6.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4568024, 1.4583867
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6450891, 1.6432238
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3305888, 1.3349905
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7729244, 1.7696056
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5524387, 1.5457153
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9123406, 0.9084406
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3559046, 1.3519418
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0450912, 2.0475640
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5274677, 1.5285888
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8372483, 1.8394465

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5832
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 5832

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7197512, upper bound: 0.7231452
time: 6.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7216234, upper bound: 0.7212730
time: 7.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4590306, 1.4568024
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6439524, 1.6450896
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3350534, 1.3305888
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7696056, 1.7741585
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5457158, 1.5536060
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9084406, 0.9131896
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3519416, 1.3576388
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0493617, 2.0450921
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5287337, 1.5274677
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8405328, 1.8372483

Time for backsubstitution: 22.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5832
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 5832

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7196449, upper bound: 0.7216233
time: 6.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7196449, upper bound: 0.7197510
time: 6.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4581656, 1.4570236
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6429806, 1.6453323
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3349686, 1.3306108
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7700310, 1.7724991
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5461178, 1.5520365
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9087331, 0.9120483
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3525348, 1.3553116
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0469489, 2.0457072
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5285425, 1.5275140
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8390718, 1.8376226

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5832
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 5832

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7213807, upper bound: 0.7215158
time: 6.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232533, upper bound: 0.7196432
time: 7.38 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 36.73 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 36.73
Output dim: 8, lower bound: -0.7196434, upper bound: 0.7232548
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 36.73
Output dim: 8, lower bound: -0.7215156, upper bound: 0.7213823
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 36.73
Output dim: 8, lower bound: -0.7197512, upper bound: 0.7231452
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 36.73
Output dim: 8, lower bound: -0.7216234, upper bound: 0.7212730
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 36.73
Output dim: 8, lower bound: -0.7196449, upper bound: 0.7216233
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 36.73
Output dim: 8, lower bound: -0.7196449, upper bound: 0.7197510
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 36.73
Output dim: 8, lower bound: -0.7213807, upper bound: 0.7215158
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 36.73
Output dim: 8, lower bound: -0.7232533, upper bound: 0.7196432

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4572659, 1.4579194
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6424379, 1.6370134
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3286796, 1.3337569
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7724571, 1.7712011
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5460577, 1.5436513
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9108517, 0.9088542
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3528419, 1.3502033
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0473061, 2.0466228
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5245075, 1.5266280
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8355889, 1.8339334

Time for backsubstitution: 21.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7196378, upper bound: 0.7206945
time: 6.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7170950, upper bound: 0.7232479
time: 6.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4574213, 1.4577641
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6400928, 1.6393585
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3294616, 1.3329744
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7724361, 1.7712226
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5484037, 1.5413060
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9113204, 0.9083855
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3512464, 1.3517990
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0471783, 2.0467510
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5257444, 1.5253911
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8335700, 1.8359518

Time for backsubstitution: 22.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7215100, upper bound: 0.7188218
time: 6.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7189672, upper bound: 0.7213762
time: 10.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4564009, 1.4581406
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6414671, 1.6372566
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3285947, 1.3337789
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7728825, 1.7695422
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5464597, 1.5420823
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9111443, 0.9077129
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3534350, 1.3478761
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0448942, 2.0472369
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5243158, 1.5266743
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8341279, 1.8343077

Time for backsubstitution: 22.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6109

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7197456, upper bound: 0.7205881
time: 4.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7172028, upper bound: 0.7231399
time: 6.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4565563, 1.4579854
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6391220, 1.6396017
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3293767, 1.3329964
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7728605, 1.7695632
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5488057, 1.5397365
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9116130, 0.9072442
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3518391, 1.3494720
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0447655, 2.0473652
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5255527, 1.5254374
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8321099, 1.8363266

Time for backsubstitution: 22.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 6109

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7216178, upper bound: 0.7187143
time: 6.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7190750, upper bound: 0.7212685
time: 5.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4586291, 1.4565563
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6403294, 1.6391220
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3330593, 1.3293767
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7695637, 1.7740951
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5397367, 1.5499725
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9072442, 0.9124618
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3494716, 1.3535733
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0491638, 2.0447659
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5255823, 1.5255527
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8374124, 1.8321095

Time for backsubstitution: 22.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 6109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7212670, upper bound: 0.7190750
time: 6.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7187141, upper bound: 0.7216182
time: 6.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4587846, 1.4564011
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6379843, 1.6414671
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3338418, 1.3285947
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7695427, 1.7741165
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5420828, 1.5476270
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9077129, 0.9119930
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3478761, 1.3551691
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0490351, 2.0448942
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5268192, 1.5243163
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8353934, 1.8341284

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 6109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7231396, upper bound: 0.7172042
time: 6.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7205866, upper bound: 0.7197454
time: 6.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4577641, 1.4567776
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6393585, 1.6393652
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3329744, 1.3293986
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7699890, 1.7724357
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5401387, 1.5484033
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9075365, 0.9113204
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3500648, 1.3512461
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0467510, 2.0453801
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5253911, 1.5255995
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8359523, 1.8324842

Time for backsubstitution: 22.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7213748, upper bound: 0.7189687
time: 5.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7188220, upper bound: 0.7215111
time: 4.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4579196, 1.4566221
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6370134, 1.6417103
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3337569, 1.3286166
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7699671, 1.7724576
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5424843, 1.5460577
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9080052, 0.9108517
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3484693, 1.3528419
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0466232, 2.0455084
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5266280, 1.5243626
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8339334, 1.8345027

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232474, upper bound: 0.7170964
time: 5.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7206944, upper bound: 0.7196392
time: 4.56 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 32.36 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7196378, upper bound: 0.7206945
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7170950, upper bound: 0.7232479
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7215100, upper bound: 0.7188218
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7189672, upper bound: 0.7213762
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7197456, upper bound: 0.7205881
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7172028, upper bound: 0.7231399
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7216178, upper bound: 0.7187143
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7190750, upper bound: 0.7212685
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7212670, upper bound: 0.7190750
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7187141, upper bound: 0.7216182
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7231396, upper bound: 0.7172042
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7205866, upper bound: 0.7197454
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7213748, upper bound: 0.7189687
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7188220, upper bound: 0.7215111
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7232474, upper bound: 0.7170964
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.36
Output dim: 8, lower bound: -0.7206944, upper bound: 0.7196392

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4550138, 1.4539139
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6413460, 1.6350718
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3272285, 1.3311810
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7653980, 1.7672272
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5291877, 1.5341613
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9079287, 0.9036484
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3500433, 1.3452306
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0281858, 2.0358906
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5205717, 1.5196171
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8307118, 1.8311944

Time for backsubstitution: 22.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6137

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7176017, upper bound: 0.7206927
time: 5.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.7196344, upper bound: 0.7186752
time: 4.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4532604, 1.4556663
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6404963, 1.6359215
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3261032, 1.3323064
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7684841, 1.7641416
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5365686, 1.5267816
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9056461, 0.9059317
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3478689, 1.3474050
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0365744, 2.0275021
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5174966, 1.5226927
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8328500, 1.8290563

Time for backsubstitution: 21.53 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 61.51 + 559.41 = 620.92 seconds
