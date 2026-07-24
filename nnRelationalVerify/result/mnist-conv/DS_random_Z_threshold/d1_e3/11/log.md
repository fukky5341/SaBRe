## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.15455141599999997


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6983652, 0.6983652)
1: (-10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5463943, 0.5463943)
2: (-8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4942775, 0.4942775)
3: (-8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4032021, 0.4032018)
4: (-3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3239938, 0.3239939)
5: (-8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4194160, 0.4194160)
6: (-13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4799125, 0.4799125)
7: (-3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4211464, 0.4211464)
8: (-0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5180578, 0.5180578)
9: (3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3047249, 0.3047249)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.63 + 33.21 = 55.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.1644164, upper bound: 0.1644164

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 538

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 538

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1625727, upper bound: 0.1644146
time: 4.26 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1644147, upper bound: 0.1625727
time: 4.43 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.71 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.71
Output dim: 9, lower bound: -0.1625727, upper bound: 0.1644146
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.71
Output dim: 9, lower bound: -0.1644147, upper bound: 0.1625727

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6946006, 0.6896887
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5450535, 0.5432954
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4932203, 0.4918456
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4001019, 0.4018574
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3225793, 0.3233795
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4193671, 0.4193947
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4765978, 0.4722726
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4184465, 0.4199758
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5134940, 0.5075326
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3012898, 0.3032352

Time for backsubstitution: 21.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1920

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1616553, upper bound: 0.1640548
time: 4.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1622132, upper bound: 0.1634974
time: 3.51 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6896892, 0.6946006
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5432954, 0.5450530
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4918456, 0.4932203
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4018574, 0.4001019
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3233794, 0.3225794
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4193947, 0.4193671
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4722726, 0.4765978
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4199758, 0.4184465
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5075326, 0.5134940
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3032353, 0.3012897

Time for backsubstitution: 20.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 724

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1731

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1641238, upper bound: 0.1612315
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1630731, upper bound: 0.1622821
time: 3.37 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.25 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.25
Output dim: 9, lower bound: -0.1616553, upper bound: 0.1640548
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.25
Output dim: 9, lower bound: -0.1622132, upper bound: 0.1634974
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.25
Output dim: 9, lower bound: -0.1641238, upper bound: 0.1612315
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.25
Output dim: 9, lower bound: -0.1630731, upper bound: 0.1622821

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6933117, 0.6885209
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5452027, 0.5437860
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4932499, 0.4920254
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4001489, 0.4019887
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3225933, 0.3233998
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4190032, 0.4186370
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4760685, 0.4718049
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4188962, 0.4201641
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5134478, 0.5074749
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3028810, 0.3049701

Time for backsubstitution: 21.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 724

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 704

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1611998, upper bound: 0.1617348
time: 4.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1593354, upper bound: 0.1635988
time: 4.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6934323, 0.6884003
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5455441, 0.5434446
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4934001, 0.4918752
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4002330, 0.4019043
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3225998, 0.3233933
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4186094, 0.4190311
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4761300, 0.4717433
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4186349, 0.4204254
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5134358, 0.5074868
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3030248, 0.3048265

Time for backsubstitution: 21.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1858

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1689

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1501484, upper bound: 0.1600058
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1587217, upper bound: 0.1514330
time: 3.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6829720, 0.6889553
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5424819, 0.5439501
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4960642, 0.4966750
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3781114, 0.3763089
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.2889735, 0.2908059
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4190857, 0.4192824
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4698055, 0.4738452
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.3827758, 0.3851709
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5012426, 0.5078402
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2876003, 0.2851431

Time for backsubstitution: 22.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 655

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3110

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1634703, upper bound: 0.1593229
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1633292, upper bound: 0.1604135
time: 3.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6840439, 0.6878834
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5421920, 0.5442395
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4953003, 0.4974389
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3780644, 0.3763559
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.2916061, 0.2881734
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4193101, 0.4190581
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4695201, 0.4741306
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.3867002, 0.3812466
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5018787, 0.5072036
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2870886, 0.2856548

Time for backsubstitution: 21.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3110

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1622574, upper bound: 0.1614873
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1611656, upper bound: 0.1616284
time: 3.57 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.10 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 9, lower bound: -0.1611998, upper bound: 0.1617348
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 9, lower bound: -0.1593354, upper bound: 0.1635988
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 9, lower bound: -0.1501484, upper bound: 0.1600058
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 9, lower bound: -0.1587217, upper bound: 0.1514330
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 9, lower bound: -0.1634703, upper bound: 0.1593229
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 9, lower bound: -0.1633292, upper bound: 0.1604135
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 9, lower bound: -0.1622574, upper bound: 0.1614873
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 9, lower bound: -0.1611656, upper bound: 0.1616284

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6904626, 0.6842866
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5450282, 0.5432892
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4933124, 0.4919567
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3984816, 0.4000807
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3225864, 0.3233573
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4030957, 0.4041684
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4690700, 0.4657674
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4202185, 0.4214072
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5109591, 0.5039115
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2981262, 0.2991025

Time for backsubstitution: 21.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1739

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1537518, upper bound: 0.1538334
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1518842, upper bound: 0.1545721
time: 3.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6891985, 0.6855512
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5450473, 0.5432701
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4933319, 0.4919372
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3983252, 0.4002371
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3225573, 0.3233864
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4041407, 0.4031234
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4700928, 0.4647446
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4198780, 0.4217477
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5098724, 0.5049977
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2971570, 0.3000717

Time for backsubstitution: 21.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 2817

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1494

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1553649, upper bound: 0.1599487
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1555769, upper bound: 0.1594656
time: 3.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6845717, 0.6867599
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5375795, 0.5320811
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4815402, 0.4742756
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3848889, 0.3734667
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3070012, 0.3140182
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4170105, 0.4192986
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4761796, 0.4718325
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4148717, 0.4101934
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5003042, 0.4990592
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2874453, 0.2979386

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2817

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1472148, upper bound: 0.1598512
time: 3.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1499935, upper bound: 0.1570703
time: 3.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6916714, 0.6796598
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5338392, 0.5358214
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4756503, 0.4801655
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3717113, 0.3866444
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3132182, 0.3078012
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4192710, 0.4170382
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4761579, 0.4718544
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4086642, 0.4164009
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5050201, 0.4943428
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2959931, 0.2893908

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1494

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1548465, upper bound: 0.1480756
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1548791, upper bound: 0.1463000
time: 3.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6895065, 0.6939335
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5390487, 0.5414524
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4909906, 0.4924145
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4005661, 0.3993533
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3213515, 0.3196068
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4181654, 0.4174664
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4712071, 0.4754078
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4196916, 0.4183397
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5074449, 0.5133677
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3018575, 0.2997646

Time for backsubstitution: 21.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1739

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1560090, upper bound: 0.1498673
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1542065, upper bound: 0.1518537
time: 3.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6890216, 0.6944180
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5396948, 0.5408068
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4910398, 0.4923654
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4011090, 0.3988106
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3204069, 0.3205514
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4174941, 0.4181378
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4710827, 0.4755323
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4198689, 0.4181623
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5074062, 0.5134058
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3017104, 0.2999120

Time for backsubstitution: 21.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 1690

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1739

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1558756, upper bound: 0.1511627
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1540093, upper bound: 0.1529580
time: 3.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6895065, 0.6939335
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5390487, 0.5414524
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4909906, 0.4924145
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4005661, 0.3993533
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3213515, 0.3196068
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4181654, 0.4174664
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4712071, 0.4754078
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4196916, 0.4183397
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5074449, 0.5133677
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3018575, 0.2997646

Time for backsubstitution: 21.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1739

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1612478, upper bound: 0.1613241
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1620941, upper bound: 0.1604777
time: 3.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6890216, 0.6944180
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5396948, 0.5408068
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4910398, 0.4923654
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4011090, 0.3988106
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3204069, 0.3205514
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4174941, 0.4181378
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4710827, 0.4755323
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4198689, 0.4181623
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5074062, 0.5134058
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3017104, 0.2999120

Time for backsubstitution: 21.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 724

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2131

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1520044, upper bound: 0.1572943
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1570668, upper bound: 0.1518773
time: 3.52 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.05 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1537518, upper bound: 0.1538334
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1518842, upper bound: 0.1545721
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1553649, upper bound: 0.1599487
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1555769, upper bound: 0.1594656
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1472148, upper bound: 0.1598512
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1499935, upper bound: 0.1570703
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1548465, upper bound: 0.1480756
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1548791, upper bound: 0.1463000
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1560090, upper bound: 0.1498673
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1542065, upper bound: 0.1518537
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1558756, upper bound: 0.1511627
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1540093, upper bound: 0.1529580
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1612478, upper bound: 0.1613241
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1620941, upper bound: 0.1604777
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1520044, upper bound: 0.1572943
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.05
Output dim: 9, lower bound: -0.1570668, upper bound: 0.1518773

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6920314, 0.6881528
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5454717, 0.5424800
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4914479, 0.4907823
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4000092, 0.4016738
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3154801, 0.3181157
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4149415, 0.4151864
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4761736, 0.4696691
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4071040, 0.4124923
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5134187, 0.5073924
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2917395, 0.2966101

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1689

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2572

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1485156, upper bound: 0.1540342
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1512925, upper bound: 0.1493539
time: 4.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6810584, 0.6697345
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5303473, 0.5270052
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4846983, 0.4847174
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3978877, 0.4000909
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3138891, 0.3122072
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4173894, 0.4166901
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4756603, 0.4713404
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4183226, 0.4198384
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5129752, 0.5071712
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2890322, 0.2911640

Time for backsubstitution: 22.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1456381, upper bound: 0.1556144
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1514864, upper bound: 0.1505381
time: 3.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6746459, 0.6761470
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5287633, 0.5285892
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4860921, 0.4833231
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3983355, 0.3996432
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3114070, 0.3146892
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4166625, 0.4174170
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4756656, 0.4713352
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4183092, 0.4198518
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5131321, 0.5070143
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2892184, 0.2909777

Time for backsubstitution: 22.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1739

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1484292, upper bound: 0.1502828
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1477330, upper bound: 0.1522158
time: 3.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6933055, 0.6890855
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5377316, 0.5309896
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4803176, 0.4875813
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3817623, 0.3961072
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3197428, 0.3201690
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4131842, 0.4020567
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4688830, 0.4534578
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4120302, 0.4173937
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5037246, 0.5056858
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2998807, 0.3024725

Time for backsubstitution: 24.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 2572

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1731

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1469302, upper bound: 0.1585252
time: 5.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1453149, upper bound: 0.1595779
time: 3.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6939974, 0.6883941
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5327477, 0.5359735
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4889560, 0.4789429
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3943517, 0.3835177
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3193690, 0.3205428
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4020290, 0.4132121
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4577830, 0.4645579
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4158640, 0.4135599
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5116467, 0.4977636
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3005269, 0.3018262

Time for backsubstitution: 24.24 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 55.84 + 559.08 = 614.92 seconds
