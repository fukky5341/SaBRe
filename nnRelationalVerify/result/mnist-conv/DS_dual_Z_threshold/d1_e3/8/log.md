## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.171128188


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7749262, 0.7749262)
1: (-13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6608329, 0.6608324)
2: (-6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4653335, 0.4653327)
3: (-8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7160082, 0.7160082)
4: (-10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4615841, 0.4615841)
5: (-8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4377449, 0.4377451)
6: (-11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5711021, 0.5711021)
7: (-12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4696124, 0.4696121)
8: (11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4565544, 0.4565542)
9: (-5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4133899, 0.4133902)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.55 + 33.78 = 56.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1746203, upper bound: 0.1746206

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 4574

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 444

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1746201, upper bound: 0.1745513
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745511, upper bound: 0.1746205
time: 3.43 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.18 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.18
Output dim: 8, lower bound: -0.1746201, upper bound: 0.1745513
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.18
Output dim: 8, lower bound: -0.1745511, upper bound: 0.1746205

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7703714, 0.7720785
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6587605, 0.6575170
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4635587, 0.4642222
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7138767, 0.7146759
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4610782, 0.4607749
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4370551, 0.4366431
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5710826, 0.5710897
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4689176, 0.4685023
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4561763, 0.4563179
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4104230, 0.4115343

Time for backsubstitution: 21.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4574

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 4574

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1744147, upper bound: 0.1745510
time: 4.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1746198, upper bound: 0.1743458
time: 4.31 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7720785, 0.7703714
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6575170, 0.6587596
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4642224, 0.4635584
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7146759, 0.7138767
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4607749, 0.4610782
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4366431, 0.4370551
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5710897, 0.5710826
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4685023, 0.4689178
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4563184, 0.4561763
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4115341, 0.4104233

Time for backsubstitution: 21.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4574

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 4574

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1743456, upper bound: 0.1746201
time: 5.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745508, upper bound: 0.1744149
time: 3.58 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.62 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.62
Output dim: 8, lower bound: -0.1744147, upper bound: 0.1745510
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.62
Output dim: 8, lower bound: -0.1746198, upper bound: 0.1743458
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.62
Output dim: 8, lower bound: -0.1743456, upper bound: 0.1746201
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.62
Output dim: 8, lower bound: -0.1745508, upper bound: 0.1744149

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7682505, 0.7691641
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6568279, 0.6548719
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4631867, 0.4636250
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7137699, 0.7145500
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4597931, 0.4592385
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4368763, 0.4364934
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5701437, 0.5696902
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4674325, 0.4667246
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4554420, 0.4557023
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4104896, 0.4116373

Time for backsubstitution: 21.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 437

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 2585

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1637924, upper bound: 0.1715979
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1714616, upper bound: 0.1639287
time: 3.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7674561, 0.7699575
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6561146, 0.6555862
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4629607, 0.4638510
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7137508, 0.7145691
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4595418, 0.4594901
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4369059, 0.4364643
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5696831, 0.5701509
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4671402, 0.4670172
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4555607, 0.4555840
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4105263, 0.4116006

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 437

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 2585

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1639976, upper bound: 0.1713927
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1716667, upper bound: 0.1637235
time: 3.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7699575, 0.7674561
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6555862, 0.6561146
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4638505, 0.4629607
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7145691, 0.7137508
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4594903, 0.4595418
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4364638, 0.4369059
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5701509, 0.5696831
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4670172, 0.4671402
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4555840, 0.4555607
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4116006, 0.4105263

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 437

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 2585

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1637233, upper bound: 0.1716669
time: 5.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1713925, upper bound: 0.1639978
time: 3.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7691641, 0.7682505
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6548719, 0.6568289
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4636250, 0.4631863
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7145500, 0.7137699
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4592385, 0.4597933
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4364934, 0.4368763
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5696902, 0.5701437
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4667244, 0.4674325
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4557023, 0.4554420
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4116373, 0.4104896

Time for backsubstitution: 21.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 437

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 2585

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1639285, upper bound: 0.1714618
time: 5.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1715977, upper bound: 0.1637926
time: 3.57 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.77 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.77
Output dim: 8, lower bound: -0.1637924, upper bound: 0.1715979
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.77
Output dim: 8, lower bound: -0.1714616, upper bound: 0.1639287
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.77
Output dim: 8, lower bound: -0.1639976, upper bound: 0.1713927
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.77
Output dim: 8, lower bound: -0.1716667, upper bound: 0.1637235
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.77
Output dim: 8, lower bound: -0.1637233, upper bound: 0.1716669
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.77
Output dim: 8, lower bound: -0.1713925, upper bound: 0.1639978
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.77
Output dim: 8, lower bound: -0.1639285, upper bound: 0.1714618
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.77
Output dim: 8, lower bound: -0.1715977, upper bound: 0.1637926

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7214527, 0.7417040
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.5812745, 0.5700665
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4401436, 0.4442415
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.5368605, 0.5044951
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4562926, 0.4563224
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.3387436, 0.3250549
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5275984, 0.5317726
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4636762, 0.4626288
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.3937869, 0.4030163
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4152405, 0.4178467

Time for backsubstitution: 21.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 437

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2523

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1592481, upper bound: 0.1692050
time: 4.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1613995, upper bound: 0.1670537
time: 4.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7407904, 0.7223663
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.5720234, 0.5793176
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4438028, 0.4405823
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.5037155, 0.5376401
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4568772, 0.4557381
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.3254377, 0.3383608
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5322256, 0.5271449
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4633367, 0.4629681
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4027557, 0.3940475
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4166987, 0.4163885

Time for backsubstitution: 21.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 437

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 2523

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1669173, upper bound: 0.1615359
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1690686, upper bound: 0.1593845
time: 3.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7206583, 0.7424974
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.5805607, 0.5707812
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4399185, 0.4444673
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.5368414, 0.5045147
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4560413, 0.4565740
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.3387731, 0.3250253
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5271378, 0.5322332
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4633839, 0.4629214
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.3939056, 0.4028976
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4152772, 0.4178100

Time for backsubstitution: 21.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 437

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 2523

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1594533, upper bound: 0.1689998
time: 5.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1616047, upper bound: 0.1668485
time: 4.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7399969, 0.7231598
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.5713096, 0.5800319
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4435778, 0.4408081
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.5036960, 0.5376596
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4566259, 0.4559896
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.3254672, 0.3383315
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5317650, 0.5276055
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4630444, 0.4632607
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4028745, 0.3939288
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4167354, 0.4163518

Time for backsubstitution: 21.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 437

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 2523

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1671225, upper bound: 0.1613307
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1692738, upper bound: 0.1591794
time: 3.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7231598, 0.7399969
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.5800323, 0.5713096
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4408083, 0.4435775
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.5376596, 0.5036960
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4559898, 0.4566257
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.3383313, 0.3254671
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5276055, 0.5317650
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4632609, 0.4630444
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.3939290, 0.4028742
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4163516, 0.4167354

Time for backsubstitution: 21.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 437

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 2523

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1591791, upper bound: 0.1692741
time: 4.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1613304, upper bound: 0.1671228
time: 4.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7424974, 0.7206583
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.5707812, 0.5805602
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4444675, 0.4399183
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.5045147, 0.5368414
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4565740, 0.4560413
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.3250254, 0.3387730
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5322332, 0.5271378
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4629214, 0.4633839
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4028978, 0.3939054
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4178097, 0.4152775

Time for backsubstitution: 21.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 437

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 2523

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1668483, upper bound: 0.1616050
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1689996, upper bound: 0.1594536
time: 3.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7223663, 0.7407904
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.5793176, 0.5720234
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4405823, 0.4438031
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.5376401, 0.5037155
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4557381, 0.4568772
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.3383609, 0.3254378
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5271449, 0.5322256
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4629681, 0.4633367
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.3940477, 0.4027560
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4163888, 0.4166989

Time for backsubstitution: 21.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 437

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 2523

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1593842, upper bound: 0.1690684
time: 6.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1615356, upper bound: 0.1669176
time: 4.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7417040, 0.7214527
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.5700665, 0.5812745
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4442415, 0.4401438
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.5044951, 0.5368605
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4563227, 0.4562929
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.3250550, 0.3387434
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5317726, 0.5275979
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4626291, 0.4636762
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4030166, 0.3937867
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4178464, 0.4152408

Time for backsubstitution: 21.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 437

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 2523

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1670534, upper bound: 0.1613997
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1692048, upper bound: 0.1592484
time: 3.84 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.60 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1592481, upper bound: 0.1692050
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1613995, upper bound: 0.1670537
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1669173, upper bound: 0.1615359
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1690686, upper bound: 0.1593845
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1594533, upper bound: 0.1689998
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1616047, upper bound: 0.1668485
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1671225, upper bound: 0.1613307
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1692738, upper bound: 0.1591794
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1591791, upper bound: 0.1692741
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1613304, upper bound: 0.1671228
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1668483, upper bound: 0.1616050
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1689996, upper bound: 0.1594536
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1593842, upper bound: 0.1690684
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1615356, upper bound: 0.1669176
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1670534, upper bound: 0.1613997
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 8, lower bound: -0.1692048, upper bound: 0.1592484

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 56.33 + 426.55 = 482.88 seconds
