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
execution time: IAR + RelationalAnalysis = 24.25 + 34.00 = 58.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.3867555, upper bound: 0.3867563

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 524

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4625

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851054, upper bound: 0.3851055
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851054, upper bound: 0.3851055
time: 3.64 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.30 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.30
Output dim: 6, lower bound: -0.3851054, upper bound: 0.3851055
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.30
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

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 524

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851015
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851017, upper bound: 0.3849670
time: 3.65 seconds

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

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 524

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849418, upper bound: 0.3849470
time: 5.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849472, upper bound: 0.3849418
time: 5.29 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 32.79 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 32.79
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851015
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 32.79
Output dim: 6, lower bound: -0.3851017, upper bound: 0.3849670
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 32.79
Output dim: 6, lower bound: -0.3849418, upper bound: 0.3849470
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 32.79
Output dim: 6, lower bound: -0.3849472, upper bound: 0.3849418

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

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3848921, upper bound: 0.3849450
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3848923, upper bound: 0.3848938
time: 3.73 seconds

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

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3848938, upper bound: 0.3848922
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849444, upper bound: 0.3848927
time: 4.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5647781, 0.5647912
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6317225, 0.6316636
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5476432, 0.5475247
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6367555, 0.6367590
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4945121, 0.4944042
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5873246, 0.5873544
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6318336, 0.6318607
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7608356, 0.7608490
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5265880, 0.5266399
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5148177, 0.5148768

Time for backsubstitution: 21.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 524

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 524

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3848921, upper bound: 0.3849450
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3848938, upper bound: 0.3848922
time: 3.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5647805, 0.5647793
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6316767, 0.6316710
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5475152, 0.5475452
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6367562, 0.6367555
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4944210, 0.4944195
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5873303, 0.5873189
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6318355, 0.6318495
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7608376, 0.7608371
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5265949, 0.5265975
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5148270, 0.5148175

Time for backsubstitution: 22.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 524

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 524

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3848923, upper bound: 0.3848938
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849444, upper bound: 0.3848927
time: 4.09 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.31 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.31
Output dim: 6, lower bound: -0.3848921, upper bound: 0.3849450
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.31
Output dim: 6, lower bound: -0.3848923, upper bound: 0.3848938
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.31
Output dim: 6, lower bound: -0.3848938, upper bound: 0.3848922
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.31
Output dim: 6, lower bound: -0.3849444, upper bound: 0.3848927
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.31
Output dim: 6, lower bound: -0.3848921, upper bound: 0.3849450
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.31
Output dim: 6, lower bound: -0.3848938, upper bound: 0.3848922
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.31
Output dim: 6, lower bound: -0.3848923, upper bound: 0.3848938
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.31
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

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2336

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1446

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3720627, upper bound: 0.3765493
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3764971, upper bound: 0.3721162
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

Time for backsubstitution: 21.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 969

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2620

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3837751, upper bound: 0.3806388
time: 3.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3806385, upper bound: 0.3837778
time: 3.60 seconds

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

Time for backsubstitution: 25.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 654

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1998

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3843966, upper bound: 0.3843083
time: 4.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3843089, upper bound: 0.3843959
time: 4.02 seconds

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

Time for backsubstitution: 21.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 969

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3817583, upper bound: 0.3848919
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849435, upper bound: 0.3817055
time: 3.77 seconds

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

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 2923

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1697

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3720233, upper bound: 0.3720360
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3720233, upper bound: 0.3720360
time: 3.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 22.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 2578

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2336

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3834471, upper bound: 0.3848922
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3848938, upper bound: 0.3834456
time: 3.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 22.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2578

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2564

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3753427, upper bound: 0.3827026
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3827010, upper bound: 0.3753439
time: 6.68 seconds

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

Time for backsubstitution: 22.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 647

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3658115, upper bound: 0.3824970
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3825486, upper bound: 0.3657603
time: 3.89 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.11 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3720627, upper bound: 0.3765493
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3764971, upper bound: 0.3721162
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3837751, upper bound: 0.3806388
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3806385, upper bound: 0.3837778
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3843966, upper bound: 0.3843083
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3843089, upper bound: 0.3843959
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3817583, upper bound: 0.3848919
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3849435, upper bound: 0.3817055
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3720233, upper bound: 0.3720360
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3720233, upper bound: 0.3720360
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3834471, upper bound: 0.3848922
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3848938, upper bound: 0.3834456
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3753427, upper bound: 0.3827026
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3827010, upper bound: 0.3753439
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3658115, upper bound: 0.3824970
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.11
Output dim: 6, lower bound: -0.3825486, upper bound: 0.3657603

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5616775, 0.5618896
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6168728, 0.6245384
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5473785, 0.5450318
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6370759, 0.6387360
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4996266, 0.5034106
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5903945, 0.5948486
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6361558, 0.6329989
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7576427, 0.7584453
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5271542, 0.5278064
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5159707, 0.5142515

Time for backsubstitution: 22.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2578

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 647

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3646343, upper bound: 0.3784481
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3813793, upper bound: 0.3605827
time: 3.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5616369, 0.5619373
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6200614, 0.6213946
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5486815, 0.5437586
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6397378, 0.6361024
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4998584, 0.5031033
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5920486, 0.5932090
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6352763, 0.6338773
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7584310, 0.7576771
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5265749, 0.5283986
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5161195, 0.5141051

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 3096

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1446

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3678050, upper bound: 0.3753714
time: 3.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3722341, upper bound: 0.3709214
time: 3.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5649967, 0.5640912
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6271665, 0.6318841
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5482404, 0.5457392
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6190774, 0.6207683
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4870584, 0.4924382
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5872400, 0.5894749
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6324096, 0.6292048
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7571969, 0.7585177
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5138984, 0.5093496
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5120516, 0.5107191

Time for backsubstitution: 22.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 969
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 2571
type: DSZ, layer: 3, pos: 3103
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2923
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 3096
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 416

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 969

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3743184, upper bound: 0.3833768
time: 3.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3835367, upper bound: 0.3738096
time: 3.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5650477, 0.5640399
1: -8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6275384, 0.6315091
2: -3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5489471, 0.5450327
3: -6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6207154, 0.6191304
4: -4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4876924, 0.4918046
5: -0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5875795, 0.5891354
6: 4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6328876, 0.6285985
7: -11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7591558, 0.7565589
8: -2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5136418, 0.5096065
9: -10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5119836, 0.5107870

Time for backsubstitution: 21.68 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.25 + 545.01 = 603.25 seconds
