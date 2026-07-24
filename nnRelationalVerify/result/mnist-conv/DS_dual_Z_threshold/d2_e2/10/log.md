## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.287744645


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6356344, 0.6356342)
1: (-7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5325336, 0.5325336)
2: (-7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5109730, 0.5109732)
3: (-5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7120657, 0.7120652)
4: (-7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6489444, 0.6489446)
5: (-0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5138854, 0.5138855)
6: (-2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6067991, 0.6067991)
7: (-10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6091228, 0.6091225)
8: (7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4724457, 0.4724457)
9: (-5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7989490, 0.7989488)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.21 + 33.88 = 56.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.3028891, upper bound: 0.3028899

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4596
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 4596

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020205, upper bound: 0.3028876
time: 4.08 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028868, upper bound: 0.3020212
time: 4.89 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.17 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.17
Output dim: 8, lower bound: -0.3020205, upper bound: 0.3028876
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.17
Output dim: 8, lower bound: -0.3028868, upper bound: 0.3020212

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6305709, 0.6318352
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5241461, 0.5263789
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5085812, 0.5096548
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7068667, 0.7082863
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6441650, 0.6425726
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5106363, 0.5094028
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6068468, 0.6068349
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6072235, 0.6082335
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4694514, 0.4702877
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7948396, 0.7933352

Time for backsubstitution: 19.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018823, upper bound: 0.3028874
time: 5.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020204, upper bound: 0.3027504
time: 5.04 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6318355, 0.6305709
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5263786, 0.5241458
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5096545, 0.5085814
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7082863, 0.7068672
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6425724, 0.6441648
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5094028, 0.5106363
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6068349, 0.6068468
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6082335, 0.6072240
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4702876, 0.4694517
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7933352, 0.7948394

Time for backsubstitution: 20.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027506, upper bound: 0.3020211
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028867, upper bound: 0.3018830
time: 3.72 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.35 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.35
Output dim: 8, lower bound: -0.3018823, upper bound: 0.3028874
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.35
Output dim: 8, lower bound: -0.3020204, upper bound: 0.3027504
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.35
Output dim: 8, lower bound: -0.3027506, upper bound: 0.3020211
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.35
Output dim: 8, lower bound: -0.3028867, upper bound: 0.3018830

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6248174, 0.6302307
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5235233, 0.5241513
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5073686, 0.5093157
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7061462, 0.7057037
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6430435, 0.6385539
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5102218, 0.5079145
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6003060, 0.6050072
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6056876, 0.6027188
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4690042, 0.4701637
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7940893, 0.7906442

Time for backsubstitution: 21.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 568

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018821, upper bound: 0.3028807
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018757, upper bound: 0.3028865
time: 4.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6289659, 0.6260815
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5219183, 0.5257566
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5082426, 0.5084417
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7042847, 0.7075648
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6401463, 0.6414521
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5091480, 0.5089889
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6050200, 0.6002941
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6017089, 0.6066978
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4693277, 0.4698402
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7921486, 0.7925851

Time for backsubstitution: 20.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 568

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020203, upper bound: 0.3027446
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020136, upper bound: 0.3027511
time: 3.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6260815, 0.6289661
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5257568, 0.5219183
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5084419, 0.5082424
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7075648, 0.7042847
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6414518, 0.6401460
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5089887, 0.5091481
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6002941, 0.6050200
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6066976, 0.6017091
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4698400, 0.4693277
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7925849, 0.7921486

Time for backsubstitution: 21.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 568

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027503, upper bound: 0.3020136
time: 4.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027439, upper bound: 0.3020209
time: 3.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6302304, 0.6248171
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5241513, 0.5235229
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5093160, 0.5073683
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7057037, 0.7061462
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6385536, 0.6430438
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5079144, 0.5102220
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6050072, 0.6003060
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6027188, 0.6056881
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4701638, 0.4690042
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7906442, 0.7940893

Time for backsubstitution: 21.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 568

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028864, upper bound: 0.3018764
time: 3.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028800, upper bound: 0.3018828
time: 4.02 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.22 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.22
Output dim: 8, lower bound: -0.3018821, upper bound: 0.3028807
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.22
Output dim: 8, lower bound: -0.3018757, upper bound: 0.3028865
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.22
Output dim: 8, lower bound: -0.3020203, upper bound: 0.3027446
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.22
Output dim: 8, lower bound: -0.3020136, upper bound: 0.3027511
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.22
Output dim: 8, lower bound: -0.3027503, upper bound: 0.3020136
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.22
Output dim: 8, lower bound: -0.3027439, upper bound: 0.3020209
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.22
Output dim: 8, lower bound: -0.3028864, upper bound: 0.3018764
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.22
Output dim: 8, lower bound: -0.3028800, upper bound: 0.3018828

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6254568, 0.6299028
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5235238, 0.5241501
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5073466, 0.5093577
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7071905, 0.7051678
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6424093, 0.6397908
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5101951, 0.5079668
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6003132, 0.6050026
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6052341, 0.6036019
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4690199, 0.4701561
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7944062, 0.7904825

Time for backsubstitution: 21.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016244, upper bound: 0.3025199
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016259, upper bound: 0.3024838
time: 4.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6244893, 0.6302307
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5235219, 0.5241513
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5073686, 0.5092940
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7056103, 0.7057037
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6430435, 0.6379194
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5102218, 0.5078876
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6003017, 0.6050072
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6056876, 0.6022651
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4689965, 0.4701637
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7939274, 0.7906442

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016178, upper bound: 0.3025264
time: 5.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016194, upper bound: 0.3024903
time: 4.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6296058, 0.6257539
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5219193, 0.5257554
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5082207, 0.5084836
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7053289, 0.7070289
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6395111, 0.6426890
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5091213, 0.5090408
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6050277, 0.6002896
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6012554, 0.6075809
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4693434, 0.4698325
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7924654, 0.7924235

Time for backsubstitution: 21.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016221, upper bound: 0.3024877
time: 6.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016585, upper bound: 0.3024861
time: 4.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6286383, 0.6260815
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5219173, 0.5257566
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5082426, 0.5084200
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7037487, 0.7075648
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6401463, 0.6408179
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5091480, 0.5089617
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6050158, 0.6002941
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6017089, 0.6062441
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4693201, 0.4698402
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7919867, 0.7925851

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016156, upper bound: 0.3024942
time: 4.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016520, upper bound: 0.3024926
time: 4.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6267214, 0.6286383
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5257578, 0.5219171
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5084200, 0.5082843
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7086091, 0.7037487
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6408176, 0.6413829
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5089616, 0.5092001
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6003013, 0.6050158
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6062441, 0.6025922
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4698558, 0.4693201
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7929022, 0.7919869

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3024919, upper bound: 0.3016527
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3024935, upper bound: 0.3016163
time: 4.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6257539, 0.6289661
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5257554, 0.5219183
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5084419, 0.5082207
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7070289, 0.7042847
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6414518, 0.6395116
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5089887, 0.5091212
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6002898, 0.6050200
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6066976, 0.6012554
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4698324, 0.4693277
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7924235, 0.7921486

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3024854, upper bound: 0.3016592
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3024870, upper bound: 0.3016228
time: 3.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6308703, 0.6244893
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5241523, 0.5235219
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5092940, 0.5074103
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7067480, 0.7056103
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6379194, 0.6442807
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5078877, 0.5102739
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6050143, 0.6003015
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6022654, 0.6065712
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4701796, 0.4689965
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7909615, 0.7939277

Time for backsubstitution: 22.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3024895, upper bound: 0.3016201
time: 4.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3025257, upper bound: 0.3016179
time: 5.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6299028, 0.6248171
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5241504, 0.5235229
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5093160, 0.5073466
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7051678, 0.7061462
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6385536, 0.6424093
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5079144, 0.5101950
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6050029, 0.6003060
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6027188, 0.6052344
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4701562, 0.4690042
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7904828, 0.7940893

Time for backsubstitution: 22.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3024830, upper bound: 0.3016267
time: 4.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3025192, upper bound: 0.3016251
time: 3.80 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.85 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3016244, upper bound: 0.3025199
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3016259, upper bound: 0.3024838
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3016178, upper bound: 0.3025264
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3016194, upper bound: 0.3024903
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3016221, upper bound: 0.3024877
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3016585, upper bound: 0.3024861
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3016156, upper bound: 0.3024942
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3016520, upper bound: 0.3024926
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3024919, upper bound: 0.3016527
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3024935, upper bound: 0.3016163
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3024854, upper bound: 0.3016592
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3024870, upper bound: 0.3016228
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3024895, upper bound: 0.3016201
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3025257, upper bound: 0.3016179
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3024830, upper bound: 0.3016267
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 8, lower bound: -0.3025192, upper bound: 0.3016251

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6254568, 0.6299043
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5235238, 0.5241506
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5073462, 0.5093586
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7071905, 0.7051659
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6424103, 0.6397896
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5101957, 0.5079668
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6003122, 0.6050034
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6052346, 0.6036019
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4690199, 0.4701562
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7944071, 0.7904820

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1467
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1795
type: DSZ, layer: 3, pos: 1115
type: DSZ, layer: 3, pos: 752
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 1969

Time for candidate selection: 0.45 seconds

### Candidate
type: DSZ, layer: 3, pos: 758

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2998489, upper bound: 0.3021574
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3012552, upper bound: 0.3006857
time: 3.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6254568, 0.6299026
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5235238, 0.5241504
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5073466, 0.5093575
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7071905, 0.7051678
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6424084, 0.6397908
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5101948, 0.5079668
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6003132, 0.6050019
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6052341, 0.6036019
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4690199, 0.4701561
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7944057, 0.7904825

Time for backsubstitution: 22.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1467
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1795
type: DSZ, layer: 3, pos: 1115
type: DSZ, layer: 3, pos: 752
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 1969

Time for candidate selection: 0.50 seconds

### Candidate
type: DSZ, layer: 3, pos: 758

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2998507, upper bound: 0.3021158
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3012568, upper bound: 0.3006405
time: 3.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6244893, 0.6302321
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5235219, 0.5241518
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5073681, 0.5092950
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7056098, 0.7057018
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6430454, 0.6379182
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5102229, 0.5078878
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6003008, 0.6050076
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6056886, 0.6022651
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4689965, 0.4701637
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7939284, 0.7906439

Time for backsubstitution: 22.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1467
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1795
type: DSZ, layer: 3, pos: 1115
type: DSZ, layer: 3, pos: 752
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 1969

Time for candidate selection: 0.37 seconds

### Candidate
type: DSZ, layer: 3, pos: 758

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2998423, upper bound: 0.3021639
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3012486, upper bound: 0.3006923
time: 3.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6244893, 0.6302307
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5235219, 0.5241516
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5073686, 0.5092938
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7056103, 0.7057037
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6430426, 0.6379194
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5102220, 0.5078876
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6003017, 0.6050062
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6056876, 0.6022651
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4689965, 0.4701636
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7939270, 0.7906442

Time for backsubstitution: 21.80 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.10 + 563.81 = 619.91 seconds
