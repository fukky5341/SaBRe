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
execution time: IAR + RelationalAnalysis = 22.84 + 33.40 = 56.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1746203, upper bound: 0.1746206

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 4574

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 444

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1746201, upper bound: 0.1745513
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745511, upper bound: 0.1746205
time: 3.33 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.86 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.86
Output dim: 8, lower bound: -0.1746201, upper bound: 0.1745513
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.86
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

Time for backsubstitution: 21.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4574

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4574

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1744147, upper bound: 0.1745510
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1746198, upper bound: 0.1743458
time: 4.09 seconds

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

Time for backsubstitution: 21.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4574

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4574

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1743456, upper bound: 0.1746201
time: 5.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745508, upper bound: 0.1744149
time: 3.50 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.65 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.65
Output dim: 8, lower bound: -0.1744147, upper bound: 0.1745510
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.65
Output dim: 8, lower bound: -0.1746198, upper bound: 0.1743458
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.65
Output dim: 8, lower bound: -0.1743456, upper bound: 0.1746201
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.65
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

Time for backsubstitution: 22.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 437
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2585

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1637924, upper bound: 0.1715979
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1714616, upper bound: 0.1639287
time: 3.74 seconds

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

Time for backsubstitution: 22.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 437
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 1844

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2832

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1742512, upper bound: 0.1738797
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1741537, upper bound: 0.1739772
time: 3.40 seconds

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

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 437
type: DSZ, layer: 3, pos: 902

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2559

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1735129, upper bound: 0.1744539
time: 4.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1741763, upper bound: 0.1737704
time: 3.61 seconds

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

Time for backsubstitution: 22.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 437

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2004

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1744651, upper bound: 0.1736573
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1737932, upper bound: 0.1743292
time: 3.21 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.10 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 8, lower bound: -0.1637924, upper bound: 0.1715979
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 8, lower bound: -0.1714616, upper bound: 0.1639287
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 8, lower bound: -0.1742512, upper bound: 0.1738797
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 8, lower bound: -0.1741537, upper bound: 0.1739772
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 8, lower bound: -0.1735129, upper bound: 0.1744539
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 8, lower bound: -0.1741763, upper bound: 0.1737704
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 8, lower bound: -0.1744651, upper bound: 0.1736573
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 8, lower bound: -0.1737932, upper bound: 0.1743292

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

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 437
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1513

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2811

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1615413, upper bound: 0.1697449
time: 4.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1619421, upper bound: 0.1693244
time: 2.97 seconds

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

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 437
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 703

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 902

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1706310, upper bound: 0.1638990
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1714318, upper bound: 0.1630982
time: 3.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7674093, 0.7698936
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6558113, 0.6555114
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4627452, 0.4633343
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7136388, 0.7144575
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4594097, 0.4594164
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4367611, 0.4363704
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5691833, 0.5699925
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4669590, 0.4667702
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4553404, 0.4552755
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4104149, 0.4114728

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 437
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 703

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 306

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1742494, upper bound: 0.1738750
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1741766, upper bound: 0.1738780
time: 3.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7674561, 0.7699099
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6561146, 0.6552825
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4624438, 0.4638510
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7136388, 0.7145691
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4595418, 0.4593580
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4368122, 0.4364643
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5695248, 0.5701509
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4671402, 0.4668360
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4555607, 0.4553638
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4103982, 0.4116006

Time for backsubstitution: 22.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 437
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 2811

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1844

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1718441, upper bound: 0.1702324
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1700855, upper bound: 0.1715702
time: 3.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7602253, 0.7576604
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6219912, 0.6286969
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4568224, 0.4533770
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7143474, 0.7133827
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4593706, 0.4594131
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4359319, 0.4363308
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5178285, 0.5239429
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4681387, 0.4673331
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4552321, 0.4552650
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.3970232, 0.3936012

Time for backsubstitution: 22.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 437
type: DSZ, layer: 3, pos: 703

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1264

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1725441, upper bound: 0.1737932
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1728522, upper bound: 0.1734851
time: 4.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7601604, 0.7576919
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6280718, 0.6225195
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4542670, 0.4558756
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7141986, 0.7135286
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4593606, 0.4594223
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4358895, 0.4363699
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5242567, 0.5173602
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4672098, 0.4682386
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4552865, 0.4552088
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.3946757, 0.3958955

Time for backsubstitution: 22.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 437
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1508

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 902

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1733458, upper bound: 0.1737407
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1741466, upper bound: 0.1729399
time: 3.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7691517, 0.7682419
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6548557, 0.6568151
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4636092, 0.4631648
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7145452, 0.7137671
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4592369, 0.4597912
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4364929, 0.4368753
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5696812, 0.5701361
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4667234, 0.4674315
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4557018, 0.4554410
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4116273, 0.4104784

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 437
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 2811

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2523

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1699169, upper bound: 0.1712590
time: 3.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1720692, upper bound: 0.1691059
time: 3.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7691555, 0.7682381
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6548586, 0.6568127
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4636035, 0.4631710
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7145462, 0.7137651
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4592364, 0.4597914
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4364929, 0.4368753
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5696831, 0.5701342
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4667234, 0.4674315
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4557018, 0.4554415
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4116263, 0.4104793

Time for backsubstitution: 21.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 437
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1508

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1682

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1646117, upper bound: 0.1651588
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1646117, upper bound: 0.1651588
time: 3.23 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.40 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1615413, upper bound: 0.1697449
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1619421, upper bound: 0.1693244
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1706310, upper bound: 0.1638990
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1714318, upper bound: 0.1630982
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1742494, upper bound: 0.1738750
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1741766, upper bound: 0.1738780
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1718441, upper bound: 0.1702324
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1700855, upper bound: 0.1715702
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1725441, upper bound: 0.1737932
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1728522, upper bound: 0.1734851
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1733458, upper bound: 0.1737407
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1741466, upper bound: 0.1729399
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1699169, upper bound: 0.1712590
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1720692, upper bound: 0.1691059
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1646117, upper bound: 0.1651588
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.40
Output dim: 8, lower bound: -0.1646117, upper bound: 0.1651588

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7685471, 0.7696390
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6568937, 0.6549535
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4632044, 0.4636495
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7138133, 0.7146091
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4597516, 0.4592531
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4369576, 0.4364877
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5686660, 0.5683208
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4674401, 0.4667430
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4551697, 0.4553840
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4104087, 0.4115670

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 437
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2377

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1682

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1622730, upper bound: 0.1539378
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1622730, upper bound: 0.1539378
time: 3.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7674618, 0.7699337
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6562586, 0.6553550
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4628978, 0.4637964
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7137656, 0.7145939
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4565835, 0.4568825
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4364753, 0.4361777
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5714808, 0.5713925
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4673676, 0.4672396
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4555826, 0.4556093
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4109848, 0.4121320

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 437
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 2559

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2377

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1732767, upper bound: 0.1736905
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1740639, upper bound: 0.1729040
time: 5.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7674170, 0.7699633
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6558838, 0.6556487
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4629064, 0.4637811
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7137656, 0.7145844
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4569340, 0.4564304
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4365783, 0.4360337
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5709243, 0.5718665
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4673338, 0.4672446
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4555826, 0.4556060
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4110100, 0.4120593

Time for backsubstitution: 21.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 437
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 703

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2879

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1737284, upper bound: 0.1734456
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1737511, upper bound: 0.1734229
time: 3.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7667742, 0.7700129
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6535039, 0.6511116
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4585891, 0.4589362
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7097487, 0.7072105
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4545188, 0.4533377
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4358478, 0.4358115
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5681410, 0.5700560
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4658308, 0.4646289
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4524822, 0.4521589
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4071865, 0.4098890

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 179
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 2879
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 902
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 1991
type: DSZ, layer: 3, pos: 437
type: DSZ, layer: 3, pos: 703
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 3123
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2004
type: DSZ, layer: 3, pos: 1417
type: DSZ, layer: 3, pos: 2585
type: DSZ, layer: 3, pos: 208
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 1682

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1713671, upper bound: 0.1691798
time: 4.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1713589, upper bound: 0.1696754
time: 5.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7675114, 0.7692757
1: -13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6516404, 0.6529756
2: -6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4580464, 0.4594789
3: -8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7063918, 0.7105675
4: -10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4533892, 0.4544673
5: -8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4362531, 0.4354062
6: -11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5695877, 0.5686092
7: -12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4647522, 0.4657078
8: 11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4521356, 0.4525056
9: -5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4088144, 0.4082611

Time for backsubstitution: 22.50 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.24 + 565.47 = 621.71 seconds
