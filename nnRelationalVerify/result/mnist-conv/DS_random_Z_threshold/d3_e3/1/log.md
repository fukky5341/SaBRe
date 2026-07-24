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
execution time: IAR + RelationalAnalysis = 22.25 + 37.10 = 59.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.7232643, upper bound: 0.7232643

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 4556
type: DSZ, layer: 1, pos: 4671
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 5832
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 822

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6137

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7212394, upper bound: 0.7232613
time: 5.75 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232610, upper bound: 0.7212409
time: 4.39 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.15 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.15
Output dim: 8, lower bound: -0.7212394, upper bound: 0.7232613
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.15
Output dim: 8, lower bound: -0.7232610, upper bound: 0.7212409

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4594951, 1.4593914
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6246815, 1.6162658
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3054543, 1.3144517
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7155981, 1.7275529
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5229838, 1.5132174
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9184086, 0.9160297
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3564100, 1.3534689
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0203972, 2.0249848
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5281296, 1.5290995
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8263240, 1.8217735

Time for backsubstitution: 21.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671
type: DSZ, layer: 1, pos: 4556
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 5832

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5830

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7211294, upper bound: 0.7232588
time: 6.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7212372, upper bound: 0.7231525
time: 4.52 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4593911, 1.4594951
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6162663, 1.6246819
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3144517, 1.3054543
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7275524, 1.7155981
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5132177, 1.5229836
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9160297, 0.9184086
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3534689, 1.3564100
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0249844, 2.0203962
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5290995, 1.5281296
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8217731, 1.8263235

Time for backsubstitution: 20.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5832
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 4671
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 4556
type: DSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5832

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7213867, upper bound: 0.7212392
time: 4.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232593, upper bound: 0.7193655
time: 5.92 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.10 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.10
Output dim: 8, lower bound: -0.7211294, upper bound: 0.7232588
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.10
Output dim: 8, lower bound: -0.7212372, upper bound: 0.7231525
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.10
Output dim: 8, lower bound: -0.7213867, upper bound: 0.7212392
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.10
Output dim: 8, lower bound: -0.7232593, upper bound: 0.7193655

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4601378, 1.4591691
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6254101, 1.6160231
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3055172, 1.3144300
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7151718, 1.7287865
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5225818, 1.5143850
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9181159, 0.9168782
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3558173, 1.3552032
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0221930, 2.0243692
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5282741, 1.5290532
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8274093, 1.8213987

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 4671
type: DSZ, layer: 1, pos: 5832
type: DSZ, layer: 1, pos: 4556
type: DSZ, layer: 1, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 822

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7121868, upper bound: 0.7231076
time: 6.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7209761, upper bound: 0.7143150
time: 7.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4592729, 1.4593914
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6244392, 1.6162658
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3054323, 1.3144517
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7155981, 1.7271271
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5229838, 1.5128155
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9184086, 0.9157369
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3564100, 1.3528759
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0197811, 2.0249848
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5280828, 1.5290995
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8259492, 1.8217735

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 4556
type: DSZ, layer: 1, pos: 4671
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 5832

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 822

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7122946, upper bound: 0.7229994
time: 9.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7210838, upper bound: 0.7142071
time: 6.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4589901, 1.4592490
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6126451, 1.6187158
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3124576, 1.3042426
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7275124, 1.7155361
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5072391, 1.5193505
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9148331, 0.9176807
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3509979, 1.3523436
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0247865, 2.0200696
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5259490, 1.5262156
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8186536, 1.8211846

Time for backsubstitution: 22.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4556
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 4671
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 822

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4556

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7197500, upper bound: 0.7212323
time: 4.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7213797, upper bound: 0.7195898
time: 5.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4591451, 1.4590938
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6103001, 1.6210608
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3132396, 1.3034604
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7274904, 1.7155576
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5095847, 1.5170047
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9153018, 0.9172120
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3494024, 1.3539393
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0246587, 2.0201983
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5271854, 1.5249786
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8166347, 1.8232036

Time for backsubstitution: 21.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 4556
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5830

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7231492, upper bound: 0.7193634
time: 6.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232570, upper bound: 0.7192569
time: 7.43 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 35.24 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 35.24
Output dim: 8, lower bound: -0.7121868, upper bound: 0.7231076
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 35.24
Output dim: 8, lower bound: -0.7209761, upper bound: 0.7143150
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 35.24
Output dim: 8, lower bound: -0.7122946, upper bound: 0.7229994
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 35.24
Output dim: 8, lower bound: -0.7210838, upper bound: 0.7142071
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 35.24
Output dim: 8, lower bound: -0.7197500, upper bound: 0.7212323
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 35.24
Output dim: 8, lower bound: -0.7213797, upper bound: 0.7195898
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 35.24
Output dim: 8, lower bound: -0.7231492, upper bound: 0.7193634
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 35.24
Output dim: 8, lower bound: -0.7232570, upper bound: 0.7192569

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4589958, 1.4586313
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6131749, 1.6005421
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3055258, 1.3144410
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7116156, 1.7258220
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5237904, 1.5158658
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9159622, 0.9142950
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3544488, 1.3514040
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0211196, 2.0239811
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5239854, 1.5254803
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8257208, 1.8199501

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4671
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 4556
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 5832

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4671

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7105102, upper bound: 0.7231052
time: 8.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7121843, upper bound: 0.7214330
time: 8.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4596004, 1.4580266
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6099286, 1.6037879
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3055286, 1.3144386
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7122087, 1.7252288
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5240626, 1.5155935
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9155328, 0.9147246
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3520179, 1.3538346
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0218053, 2.0232949
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5247016, 1.5247641
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8259611, 1.8197098

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5832
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 4671
type: DSZ, layer: 1, pos: 4556
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5832

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.7191021, upper bound: 0.7143117
time: 6.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7209745, upper bound: 0.7124406
time: 6.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4581304, 1.4588528
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6122041, 1.6007853
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3054409, 1.3144634
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7120409, 1.7241631
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5241919, 1.5142965
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9162548, 0.9131539
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3550415, 1.3490767
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0187068, 2.0245962
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5237942, 1.5255265
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8242598, 1.8203249

Time for backsubstitution: 22.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5832
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 4556
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5832

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7104203, upper bound: 0.7229982
time: 8.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7122929, upper bound: 0.7211260
time: 6.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4587350, 1.4582481
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6089578, 1.6040311
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3054438, 1.3144610
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7126341, 1.7235699
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5244646, 1.5140238
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9158254, 0.9135833
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3526111, 1.3515074
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0193925, 2.0239100
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5245104, 1.5248103
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8245001, 1.8200846

Time for backsubstitution: 21.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5832
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 4556
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5832

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.7192099, upper bound: 0.7142040
time: 6.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7210821, upper bound: 0.7123312
time: 5.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4572082, 1.4588304
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6160889, 1.6200514
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.2941642, 1.2903290
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7129917, 1.6981201
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.4907529, 1.4965436
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.8968298, 0.8960698
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3357644, 1.3337395
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0225201, 2.0196605
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5195045, 1.5208464
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8076878, 1.8120437

Time for backsubstitution: 21.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 4671
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6124

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7183728, upper bound: 0.7212318
time: 4.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7197494, upper bound: 0.7198467
time: 5.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4585710, 1.4574671
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6139793, 1.6221609
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.2985415, 1.2859490
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7100964, 1.7010145
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.4844320, 1.5028617
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.8932223, 0.8996756
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3323941, 1.3371093
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0243769, 2.0178032
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5205793, 1.5197716
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8095121, 1.8102193

Time for backsubstitution: 20.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 4671
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6127

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.7188195, upper bound: 0.7195858
time: 7.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7213754, upper bound: 0.7170380
time: 5.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4597878, 1.4588716
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6110268, 1.6208167
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3133035, 1.3034389
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7270651, 1.7167912
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5091827, 1.5181720
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9150088, 0.9180603
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3488092, 1.3556733
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0264568, 2.0195837
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5273299, 1.5249319
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8177214, 1.8228288

Time for backsubstitution: 20.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4671
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 4556
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 6124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4671

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7214731, upper bound: 0.7193623
time: 4.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7231467, upper bound: 0.7176844
time: 5.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4589229, 1.4590938
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6100550, 1.6210608
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3132186, 1.3034604
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7274904, 1.7151318
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5095847, 1.5166025
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9153018, 0.9169192
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3494024, 1.3533463
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0240440, 2.0201983
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5271387, 1.5249786
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8162603, 1.8232036

Time for backsubstitution: 20.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 4556
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 4671

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232511, upper bound: 0.7167016
time: 4.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7206982, upper bound: 0.7192510
time: 7.85 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 33.21 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7105102, upper bound: 0.7231052
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7121843, upper bound: 0.7214330
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7191021, upper bound: 0.7143117
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7209745, upper bound: 0.7124406
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7104203, upper bound: 0.7229982
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7122929, upper bound: 0.7211260
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7192099, upper bound: 0.7142040
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7210821, upper bound: 0.7123312
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7183728, upper bound: 0.7212318
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7197494, upper bound: 0.7198467
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7188195, upper bound: 0.7195858
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7213754, upper bound: 0.7170380
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7214731, upper bound: 0.7193623
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7231467, upper bound: 0.7176844
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7232511, upper bound: 0.7167016
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 8, lower bound: -0.7206982, upper bound: 0.7192510

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4528222, 1.4482973
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6028881, 1.5832992
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.2942004, 1.2954786
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7021437, 1.7201719
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5207253, 1.5107396
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9091003, 0.9102035
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3492827, 1.3427515
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0190492, 2.0205221
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5187259, 1.5223408
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8255539, 1.8198476

Time for backsubstitution: 20.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4556
type: DSZ, layer: 1, pos: 6109
type: DSZ, layer: 1, pos: 6124
type: DSZ, layer: 1, pos: 5832
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4556

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7088602, upper bound: 0.7230983
time: 5.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7105029, upper bound: 0.7214674
time: 4.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4486613, 1.4524579
1: -10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.5959320, 1.5902553
2: -4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.2865634, 1.3031156
3: -5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7059641, 1.7163510
4: -13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5186639, 1.5128009
5: -3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9118705, 0.9074330
6: -10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3457961, 1.3462377
7: -9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0176606, 2.0219116
8: 9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5208459, 1.5202208
9: -7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8256178, 1.8197832

Time for backsubstitution: 21.04 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 59.36 + 541.47 = 600.82 seconds
