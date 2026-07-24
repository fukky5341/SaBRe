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
execution time: IAR + RelationalAnalysis = 21.99 + 33.61 = 55.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.1644164, upper bound: 0.1644164

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 538

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 538

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1625727, upper bound: 0.1644146
time: 4.27 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1644147, upper bound: 0.1625727
time: 4.47 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.90 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.90
Output dim: 9, lower bound: -0.1625727, upper bound: 0.1644146
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.90
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

Time for backsubstitution: 20.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1596412, upper bound: 0.1642680
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1624247, upper bound: 0.1614830
time: 4.12 seconds

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

Time for backsubstitution: 20.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1614833, upper bound: 0.1624244
time: 4.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1642683, upper bound: 0.1596409
time: 3.82 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.48 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.48
Output dim: 9, lower bound: -0.1596412, upper bound: 0.1642680
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.48
Output dim: 9, lower bound: -0.1624247, upper bound: 0.1614830
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.48
Output dim: 9, lower bound: -0.1614833, upper bound: 0.1624244
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.48
Output dim: 9, lower bound: -0.1642683, upper bound: 0.1596409

## BFS DS instance: DS_DSZ1_DSZ1

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

Time for backsubstitution: 21.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 704

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1591650, upper bound: 0.1619527
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1579190, upper bound: 0.1638105
time: 3.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2

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

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 704

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1619683, upper bound: 0.1597607
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1601099, upper bound: 0.1610068
time: 4.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6883941, 0.6939969
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5359735, 0.5327477
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4789429, 0.4889560
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3835177, 0.3943517
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3205429, 0.3193688
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4132121, 0.4020290
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4645579, 0.4577830
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4135599, 0.4158640
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.4977636, 0.5116467
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3018262, 0.3005270

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 704

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1610074, upper bound: 0.1601099
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1597607, upper bound: 0.1619679
time: 3.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6890855, 0.6933055
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5309896, 0.5377312
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4875813, 0.4803176
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3961072, 0.3817620
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3201691, 0.3197427
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4020567, 0.4131842
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4534578, 0.4688830
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4173937, 0.4120302
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5056853, 0.5037251
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3024724, 0.2998807

Time for backsubstitution: 21.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 704

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1638107, upper bound: 0.1579186
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1619531, upper bound: 0.1591650
time: 3.58 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.76 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.76
Output dim: 9, lower bound: -0.1591650, upper bound: 0.1619527
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.76
Output dim: 9, lower bound: -0.1579190, upper bound: 0.1638105
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.76
Output dim: 9, lower bound: -0.1619683, upper bound: 0.1597607
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.76
Output dim: 9, lower bound: -0.1601099, upper bound: 0.1610068
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.76
Output dim: 9, lower bound: -0.1610074, upper bound: 0.1601099
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.76
Output dim: 9, lower bound: -0.1597607, upper bound: 0.1619679
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.76
Output dim: 9, lower bound: -0.1638107, upper bound: 0.1579186
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.76
Output dim: 9, lower bound: -0.1619531, upper bound: 0.1591650

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

Time for backsubstitution: 21.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Candidate
type: DSZ, layer: 3, pos: 1942

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1563567, upper bound: 0.1607315
time: 3.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1580541, upper bound: 0.1594312
time: 3.50 seconds

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

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Candidate
type: DSZ, layer: 3, pos: 1942

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1551099, upper bound: 0.1625883
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1568068, upper bound: 0.1612880
time: 4.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Candidate
type: DSZ, layer: 3, pos: 1942

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1591521, upper bound: 0.1585482
time: 4.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1608477, upper bound: 0.1572460
time: 5.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Candidate
type: DSZ, layer: 3, pos: 1942

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1572952, upper bound: 0.1597954
time: 8.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1589909, upper bound: 0.1584929
time: 3.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6855512, 0.6891985
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5432701, 0.5450473
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4919372, 0.4933314
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4002373, 0.3983250
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3233865, 0.3225574
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4031234, 0.4041407
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4647446, 0.4700925
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4217477, 0.4198780
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5049977, 0.5098724
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3000717, 0.2971570

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Candidate
type: DSZ, layer: 3, pos: 1942

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1584932, upper bound: 0.1589906
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1597957, upper bound: 0.1572950
time: 3.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6842866, 0.6904626
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5432892, 0.5450282
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4919567, 0.4933119
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4000807, 0.3984816
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3233575, 0.3225865
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4041684, 0.4030957
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4657674, 0.4690697
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4214072, 0.4202185
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5039115, 0.5109591
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2991025, 0.2981262

Time for backsubstitution: 21.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Candidate
type: DSZ, layer: 3, pos: 1942

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1572466, upper bound: 0.1608475
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1585487, upper bound: 0.1591518
time: 3.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6855512, 0.6891985
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5432701, 0.5450473
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4919372, 0.4933314
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4002373, 0.3983250
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3233865, 0.3225574
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4031234, 0.4041407
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4647446, 0.4700925
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4217477, 0.4198780
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5049977, 0.5098724
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3000717, 0.2971570

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Candidate
type: DSZ, layer: 3, pos: 1942

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1612882, upper bound: 0.1568065
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1625887, upper bound: 0.1551095
time: 3.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6842866, 0.6904626
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5432892, 0.5450282
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4919567, 0.4933119
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4000807, 0.3984816
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3233575, 0.3225865
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4041684, 0.4030957
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4657674, 0.4690697
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4214072, 0.4202185
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5039115, 0.5109591
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2991025, 0.2981262

Time for backsubstitution: 21.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Candidate
type: DSZ, layer: 3, pos: 1942

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1594315, upper bound: 0.1580538
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1607318, upper bound: 0.1563565
time: 3.29 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1563567, upper bound: 0.1607315
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1580541, upper bound: 0.1594312
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1551099, upper bound: 0.1625883
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1568068, upper bound: 0.1612880
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1591521, upper bound: 0.1585482
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1608477, upper bound: 0.1572460
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1572952, upper bound: 0.1597954
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1589909, upper bound: 0.1584929
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1584932, upper bound: 0.1589906
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1597957, upper bound: 0.1572950
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1572466, upper bound: 0.1608475
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1585487, upper bound: 0.1591518
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1612882, upper bound: 0.1568065
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1625887, upper bound: 0.1551095
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1594315, upper bound: 0.1580538
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 9, lower bound: -0.1607318, upper bound: 0.1563565

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6877384, 0.6820779
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5334892, 0.5301290
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4880590, 0.4878330
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3993285, 0.4010513
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3205816, 0.3212280
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4026511, 0.3994083
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4763913, 0.4720571
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.3994107, 0.4047275
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.4763198, 0.4635081
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2975833, 0.2994363

Time for backsubstitution: 21.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Candidate
type: DSZ, layer: 3, pos: 704

### Candidate
type: DSZ, layer: 3, pos: 2131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1464146, upper bound: 0.1567446
time: 4.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1521283, upper bound: 0.1511520
time: 4.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6869898, 0.6832547
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5322065, 0.5317316
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4892077, 0.4869232
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3992956, 0.4011250
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3205036, 0.3213816
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.3993807, 0.4031999
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4763823, 0.4720774
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4037104, 0.4009399
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.4694691, 0.4715776
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2976220, 0.2995288

Time for backsubstitution: 21.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Candidate
type: DSZ, layer: 3, pos: 704

### Candidate
type: DSZ, layer: 3, pos: 2131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1479519, upper bound: 0.1553635
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1539094, upper bound: 0.1499826
time: 3.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6877384, 0.6820779
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5334892, 0.5301290
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4880590, 0.4878330
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3993285, 0.4010513
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3205816, 0.3212280
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4026511, 0.3994083
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4763913, 0.4720571
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.3994107, 0.4047275
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.4763198, 0.4635081
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2975833, 0.2994363

Time for backsubstitution: 22.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Candidate
type: DSZ, layer: 3, pos: 704

### Candidate
type: DSZ, layer: 3, pos: 2131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1455946, upper bound: 0.1582758
time: 3.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1510456, upper bound: 0.1529251
time: 4.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6869898, 0.6832547
1: -10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5322065, 0.5317316
2: -8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4892077, 0.4869232
3: -8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.3992956, 0.4011250
4: -3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3205036, 0.3213816
5: -8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.3993807, 0.4031999
6: -13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4763823, 0.4720774
7: -3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4037104, 0.4009399
8: -0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.4694691, 0.4715776
9: 3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.2976220, 0.2995288

Time for backsubstitution: 21.51 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 655
type: DSZ, layer: 3, pos: 704
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 3110
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 655

### Candidate
type: DSZ, layer: 3, pos: 704

### Candidate
type: DSZ, layer: 3, pos: 2131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1471608, upper bound: 0.1568945
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1528260, upper bound: 0.1517778
time: 3.84 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 29.08 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 29.08
Output dim: 9, lower bound: -0.1464146, upper bound: 0.1567446
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.08
Output dim: 9, lower bound: -0.1521283, upper bound: 0.1511520
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 29.08
Output dim: 9, lower bound: -0.1479519, upper bound: 0.1553635
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.08
Output dim: 9, lower bound: -0.1539094, upper bound: 0.1499826
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 29.08
Output dim: 9, lower bound: -0.1455946, upper bound: 0.1582758
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.08
Output dim: 9, lower bound: -0.1510456, upper bound: 0.1529251
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 29.08
Output dim: 9, lower bound: -0.1471608, upper bound: 0.1568945
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.08
Output dim: 9, lower bound: -0.1528260, upper bound: 0.1517778
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.08
Output dim: 9, lower bound: -0.1591521, upper bound: 0.1585482
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.08
Output dim: 9, lower bound: -0.1608477, upper bound: 0.1572460
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.08
Output dim: 9, lower bound: -0.1572952, upper bound: 0.1597954
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.08
Output dim: 9, lower bound: -0.1589909, upper bound: 0.1584929
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.08
Output dim: 9, lower bound: -0.1584932, upper bound: 0.1589906
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.08
Output dim: 9, lower bound: -0.1597957, upper bound: 0.1572950
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.08
Output dim: 9, lower bound: -0.1572466, upper bound: 0.1608475
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.08
Output dim: 9, lower bound: -0.1585487, upper bound: 0.1591518
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.08
Output dim: 9, lower bound: -0.1612882, upper bound: 0.1568065
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.08
Output dim: 9, lower bound: -0.1625887, upper bound: 0.1551095
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.08
Output dim: 9, lower bound: -0.1594315, upper bound: 0.1580538
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.08
Output dim: 9, lower bound: -0.1607318, upper bound: 0.1563565

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 55.60 + 544.73 = 600.33 seconds
