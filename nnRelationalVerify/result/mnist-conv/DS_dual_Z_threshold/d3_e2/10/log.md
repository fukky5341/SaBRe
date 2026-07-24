## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.37902039


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5647817, 0.5647817)
1: (-8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6316841, 0.6316841)
2: (-3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5475452, 0.5475452)
3: (-6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6367562, 0.6367562)
4: (-4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4944360, 0.4944361)
5: (-0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5873303, 0.5873306)
6: (4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6318514, 0.6318514)
7: (-11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7608390, 0.7608390)
8: (-2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5266042, 0.5266042)
9: (-10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5148268, 0.5148268)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.99 + 33.94 = 56.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.3867555, upper bound: 0.3867563

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 4625

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851054, upper bound: 0.3851055
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851054, upper bound: 0.3851055
time: 3.92 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.01 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.01
Output dim: 6, lower bound: -0.3851054, upper bound: 0.3851055
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.01
Output dim: 6, lower bound: -0.3851054, upper bound: 0.3851055

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5655098, 0.5647805
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6316710, 0.6360838
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5507252, 0.5475364
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6369119, 0.6367562
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4944193, 0.4989997
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5873246, 0.5893686
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6354134, 0.6318355
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7613535, 0.7608376
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5298448, 0.5265949
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5162549, 0.5148270

Time for backsubstitution: 22.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 524

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849676, upper bound: 0.3851011
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851017, upper bound: 0.3849670
time: 4.50 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5647805, 0.5647817
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6316841, 0.6316710
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5475364, 0.5475452
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6367562, 0.6367562
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4944360, 0.4944195
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5873303, 0.5873249
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6318355, 0.6318514
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7608376, 0.7608390
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5265949, 0.5266042
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5148270, 0.5148268

Time for backsubstitution: 22.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 524

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851015
time: 4.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851017, upper bound: 0.3849670
time: 4.21 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.38 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.38
Output dim: 6, lower bound: -0.3849676, upper bound: 0.3851011
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.38
Output dim: 6, lower bound: -0.3851017, upper bound: 0.3849670
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.38
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851015
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.38
Output dim: 6, lower bound: -0.3851017, upper bound: 0.3849670

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5652702, 0.5655320
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6316462, 0.6361606
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5510678, 0.5474269
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6375515, 0.6365509
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4952736, 0.4987279
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5871410, 0.5899465
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6350985, 0.6328256
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7612200, 0.7612562
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5287600, 0.5300108
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5165794, 0.5147235

Time for backsubstitution: 22.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3848921, upper bound: 0.3849450
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3848923, upper bound: 0.3848938
time: 4.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5655098, 0.5645409
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6316710, 0.6360590
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5506160, 0.5475364
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6367066, 0.6367562
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4941483, 0.4989997
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5873246, 0.5891848
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6354134, 0.6315207
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7613535, 0.7607040
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5298448, 0.5255101
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5161512, 0.5148270

Time for backsubstitution: 22.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3848938, upper bound: 0.3848922
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849444, upper bound: 0.3848927
time: 4.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5645409, 0.5655334
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6316590, 0.6317475
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5478790, 0.5474365
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6373959, 0.6365511
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4952903, 0.4941480
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5871463, 0.5879030
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6315207, 0.6328416
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7607036, 0.7612581
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5255101, 0.5300202
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5151515, 0.5147231

Time for backsubstitution: 22.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3848921, upper bound: 0.3849450
time: 4.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3848923, upper bound: 0.3848938
time: 4.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5647805, 0.5645421
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6316841, 0.6316462
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5474269, 0.5475452
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6365509, 0.6367562
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4941645, 0.4944195
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5873303, 0.5871408
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6318355, 0.6315365
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7608376, 0.7607055
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5265949, 0.5255195
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5147235, 0.5148268

Time for backsubstitution: 22.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3848938, upper bound: 0.3848922
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849444, upper bound: 0.3848927
time: 4.32 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.47 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.47
Output dim: 6, lower bound: -0.3848921, upper bound: 0.3849450
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.47
Output dim: 6, lower bound: -0.3848923, upper bound: 0.3848938
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.47
Output dim: 6, lower bound: -0.3848938, upper bound: 0.3848922
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.47
Output dim: 6, lower bound: -0.3849444, upper bound: 0.3848927
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.47
Output dim: 6, lower bound: -0.3848921, upper bound: 0.3849450
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.47
Output dim: 6, lower bound: -0.3848923, upper bound: 0.3848938
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.47
Output dim: 6, lower bound: -0.3848938, upper bound: 0.3848922
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.47
Output dim: 6, lower bound: -0.3849444, upper bound: 0.3848927

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5652676, 0.5655446
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6316845, 0.6361532
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5511751, 0.5474064
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6375508, 0.6365540
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4953520, 0.4987124
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5871348, 0.5899765
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6350968, 0.6328351
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7612185, 0.7612667
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5287530, 0.5300465
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5165699, 0.5147736

Time for backsubstitution: 22.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 647

Time for candidate selection: 0.61 seconds

### Candidate
type: DSZ, layer: 3, pos: 2578

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3639668, upper bound: 0.3813959
time: 3.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3813772, upper bound: 0.3640724
time: 3.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5652702, 0.5655296
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6316388, 0.6361606
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5510471, 0.5474269
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6375515, 0.6365504
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4952605, 0.4987279
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5871410, 0.5899405
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6350985, 0.6328239
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7612200, 0.7612548
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5287600, 0.5300040
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5165794, 0.5147142

Time for backsubstitution: 22.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 647

Time for candidate selection: 0.59 seconds

### Candidate
type: DSZ, layer: 3, pos: 2578

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3639675, upper bound: 0.3813781
time: 3.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3813772, upper bound: 0.3639719
time: 3.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5655074, 0.5645504
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6317096, 0.6360517
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5507231, 0.5475152
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6367059, 0.6367593
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4942262, 0.4989842
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5873189, 0.5892141
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6354117, 0.6315303
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7613516, 0.7607145
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5298378, 0.5255457
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5161419, 0.5148773

Time for backsubstitution: 22.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 647

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 3, pos: 2578

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3639712, upper bound: 0.3813777
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3813776, upper bound: 0.3639682
time: 3.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5655098, 0.5645382
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6316636, 0.6360590
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5505953, 0.5475364
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6367066, 0.6367557
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4941351, 0.4989997
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5873246, 0.5891786
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6354134, 0.6315188
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7613535, 0.7607021
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5298448, 0.5255033
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5161512, 0.5148177

Time for backsubstitution: 22.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 647

Time for candidate selection: 0.48 seconds

### Candidate
type: DSZ, layer: 3, pos: 2578

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3640718, upper bound: 0.3813776
time: 4.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3813953, upper bound: 0.3639675
time: 3.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5645382, 0.5655458
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6316974, 0.6317403
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5479863, 0.5474157
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6373951, 0.6365540
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4953687, 0.4941349
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5871406, 0.5879331
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6315188, 0.6328511
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7607021, 0.7612686
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5255032, 0.5300560
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5151422, 0.5147734

Time for backsubstitution: 22.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 647

Time for candidate selection: 0.49 seconds

### Candidate
type: DSZ, layer: 3, pos: 2578

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3639668, upper bound: 0.3813959
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3813772, upper bound: 0.3640724
time: 3.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5645409, 0.5655308
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6316516, 0.6317475
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5478582, 0.5474365
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6373959, 0.6365504
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4952772, 0.4941480
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5871463, 0.5878968
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6315207, 0.6328397
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7607036, 0.7612567
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5255101, 0.5300136
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5151515, 0.5147138

Time for backsubstitution: 22.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 647

Time for candidate selection: 0.46 seconds

### Candidate
type: DSZ, layer: 3, pos: 2578

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3639675, upper bound: 0.3813781
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3813772, upper bound: 0.3639719
time: 3.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5647781, 0.5645516
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6317225, 0.6316388
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5475342, 0.5475247
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6365504, 0.6367590
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4942424, 0.4944042
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5873246, 0.5871704
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6318336, 0.6315460
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7608356, 0.7607160
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5265880, 0.5255551
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5147142, 0.5148768

Time for backsubstitution: 22.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 647

Time for candidate selection: 0.57 seconds

### Candidate
type: DSZ, layer: 3, pos: 2578

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3639712, upper bound: 0.3813777
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3813776, upper bound: 0.3639682
time: 3.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5647805, 0.5645394
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6316767, 0.6316462
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5474064, 0.5475452
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6365509, 0.6367555
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4941518, 0.4944195
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5873303, 0.5871348
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6318355, 0.6315346
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7608376, 0.7607040
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5265949, 0.5255127
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5147235, 0.5148175

Time for backsubstitution: 22.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 647

Time for candidate selection: 0.58 seconds

### Candidate
type: DSZ, layer: 3, pos: 2578

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3640718, upper bound: 0.3813776
time: 4.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3813953, upper bound: 0.3639675
time: 3.97 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.81 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3639668, upper bound: 0.3813959
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3813772, upper bound: 0.3640724
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3639675, upper bound: 0.3813781
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3813772, upper bound: 0.3639719
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3639712, upper bound: 0.3813777
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3813776, upper bound: 0.3639682
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3640718, upper bound: 0.3813776
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3813953, upper bound: 0.3639675
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3639668, upper bound: 0.3813959
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3813772, upper bound: 0.3640724
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3639675, upper bound: 0.3813781
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3813772, upper bound: 0.3639719
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3639712, upper bound: 0.3813777
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3813776, upper bound: 0.3639682
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3640718, upper bound: 0.3813776
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.81
Output dim: 6, lower bound: -0.3813953, upper bound: 0.3639675

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5610018, 0.5594132
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.5648316, 0.5607986
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5053413, 0.5098939
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6005359, 0.6017374
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4978917, 0.5004951
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5443170, 0.5325232
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.5857050, 0.5893648
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7230024, 0.7181730
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5316312, 0.5383239
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.4810290, 0.4806519

Time for backsubstitution: 22.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 647

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2564

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3544332, upper bound: 0.3792207
time: 4.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3617880, upper bound: 0.3718171
time: 3.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5606861, 0.5612791
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.5563301, 0.5716126
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5162201, 0.5015726
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6027343, 0.6018863
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4971216, 0.5013870
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5296812, 0.5491707
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.5936792, 0.5834692
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7181244, 0.7262583
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5390301, 0.5329247
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.4824479, 0.4814074

Time for backsubstitution: 22.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 647

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2564

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3717974, upper bound: 0.3618896
time: 5.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3792021, upper bound: 0.3545356
time: 3.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5610044, 0.5593977
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.5647858, 0.5608060
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5052135, 0.5099151
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6005361, 0.6017337
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4978004, 0.5005106
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5443230, 0.5324869
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.5857069, 0.5893536
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7230053, 0.7181606
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5316362, 0.5382814
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.4810386, 0.4805923

Time for backsubstitution: 22.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 647

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2564

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3544340, upper bound: 0.3792027
time: 4.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3617888, upper bound: 0.3717985
time: 3.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5606887, 0.5612638
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.5562843, 0.5716202
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5160944, 0.5015938
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6027346, 0.6018826
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4970303, 0.5014025
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5296872, 0.5491648
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.5936811, 0.5834579
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7181263, 0.7262468
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5390351, 0.5328822
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.4824576, 0.4813478

Time for backsubstitution: 22.46 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.93 + 557.14 = 614.07 seconds
