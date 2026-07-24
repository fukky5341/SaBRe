## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.6088244805


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.1174488, -10.4751902, -13.1174488, -10.4751902, -1.8021035, 1.8021038)
1: (-7.1292858, -4.1849318, -7.1292858, -4.1849318, -2.3330483, 2.3330483)
2: (9.3677406, 11.2813492, 9.3677406, 11.2813492, -1.5663891, 1.5663891)
3: (-4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9216013, 1.9216008)
4: (-9.4387360, -6.7248478, -9.4387360, -6.7248478, -1.9660249, 1.9660249)
5: (-13.7978468, -11.1748791, -13.7978468, -11.1748791, -1.6303473, 1.6303473)
6: (-16.3375626, -12.7550831, -16.3375626, -12.7550831, -2.2865324, 2.2865324)
7: (-4.0563126, -1.3696806, -4.0563126, -1.3696806, -2.4632163, 2.4632158)
8: (-6.0375504, -3.6194944, -6.0375504, -3.6194944, -2.0286188, 2.0286188)
9: (-11.8428965, -9.3279085, -11.8428965, -9.3279085, -1.7399156, 1.7399158)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.89 + 37.98 = 61.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.6118839, upper bound: 0.6118833

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 6198
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 4586
type: DSZ, layer: 1, pos: 6218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5735

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118729, upper bound: 0.6062237
time: 9.25 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6062216, upper bound: 0.6118723
time: 12.04 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 21.31 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 21.31
Output dim: 2, lower bound: -0.6118729, upper bound: 0.6062237
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 21.31
Output dim: 2, lower bound: -0.6062216, upper bound: 0.6118723

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -13.1174488, -10.4751902, -13.1174488, -10.4751902, -1.7782464, 1.7889299
1: -7.1292858, -4.1849318, -7.1292858, -4.1849318, -2.3244457, 2.3174591
2: 9.3677406, 11.2813492, 9.3677406, 11.2813492, -1.5590239, 1.5530324
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9176044, 1.9143510
4: -9.4387360, -6.7248478, -9.4387360, -6.7248478, -1.9613333, 1.9634380
5: -13.7978468, -11.1748791, -13.7978468, -11.1748791, -1.6161118, 1.6224988
6: -16.3375626, -12.7550831, -16.3375626, -12.7550831, -2.2850213, 2.2837901
7: -4.0563126, -1.3696806, -4.0563126, -1.3696806, -2.4624057, 2.4627686
8: -6.0375504, -3.6194944, -6.0375504, -3.6194944, -2.0258746, 2.0271072
9: -11.8428965, -9.3279085, -11.8428965, -9.3279085, -1.7239554, 1.7311115

Time for backsubstitution: 22.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 4586
type: DSZ, layer: 1, pos: 6198
type: DSZ, layer: 1, pos: 6218
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5762

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6085120, upper bound: 0.6062178
time: 8.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118697, upper bound: 0.6028628
time: 5.01 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -13.1174488, -10.4751902, -13.1174488, -10.4751902, -1.7889295, 1.7782471
1: -7.1292858, -4.1849318, -7.1292858, -4.1849318, -2.3174591, 2.3244452
2: 9.3677406, 11.2813492, 9.3677406, 11.2813492, -1.5530324, 1.5590239
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9143505, 1.9176049
4: -9.4387360, -6.7248478, -9.4387360, -6.7248478, -1.9634380, 1.9613333
5: -13.7978468, -11.1748791, -13.7978468, -11.1748791, -1.6224985, 1.6161120
6: -16.3375626, -12.7550831, -16.3375626, -12.7550831, -2.2837901, 2.2850213
7: -4.0563126, -1.3696806, -4.0563126, -1.3696806, -2.4627690, 2.4624062
8: -6.0375504, -3.6194944, -6.0375504, -3.6194944, -2.0271072, 2.0258746
9: -11.8428965, -9.3279085, -11.8428965, -9.3279085, -1.7311118, 1.7239552

Time for backsubstitution: 21.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6198
type: DSZ, layer: 1, pos: 6218
type: DSZ, layer: 1, pos: 4586
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6198

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6031187, upper bound: 0.6118660
time: 6.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6062151, upper bound: 0.6087722
time: 5.61 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 33.42 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 33.42
Output dim: 2, lower bound: -0.6085120, upper bound: 0.6062178
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 33.42
Output dim: 2, lower bound: -0.6118697, upper bound: 0.6028628
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 33.42
Output dim: 2, lower bound: -0.6031187, upper bound: 0.6118660
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 33.42
Output dim: 2, lower bound: -0.6062151, upper bound: 0.6087722

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.1174488, -10.4751902, -13.1174488, -10.4751902, -1.7651200, 1.7825797
1: -7.1292858, -4.1849318, -7.1292858, -4.1849318, -2.3153725, 2.3130736
2: 9.3677406, 11.2813492, 9.3677406, 11.2813492, -1.5550528, 1.5448170
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9143119, 1.9127545
4: -9.4387360, -6.7248478, -9.4387360, -6.7248478, -1.9547186, 1.9602332
5: -13.7978468, -11.1748791, -13.7978468, -11.1748791, -1.6097789, 1.6194291
6: -16.3375626, -12.7550831, -16.3375626, -12.7550831, -2.2803001, 2.2814982
7: -4.0563126, -1.3696806, -4.0563126, -1.3696806, -2.4581251, 2.4606957
8: -6.0375504, -3.6194944, -6.0375504, -3.6194944, -2.0191255, 2.0238423
9: -11.8428965, -9.3279085, -11.8428965, -9.3279085, -1.7227786, 1.7305422

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6198
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 6218
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 4586

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6198

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6087668, upper bound: 0.6028562
time: 14.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118632, upper bound: 0.5997599
time: 4.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.1174488, -10.4751902, -13.1174488, -10.4751902, -1.7881589, 1.7776866
1: -7.1292858, -4.1849318, -7.1292858, -4.1849318, -2.3166199, 2.3238368
2: 9.3677406, 11.2813492, 9.3677406, 11.2813492, -1.5403986, 1.5498598
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9122353, 1.9160705
4: -9.4387360, -6.7248478, -9.4387360, -6.7248478, -1.9625812, 1.9607115
5: -13.7978468, -11.1748791, -13.7978468, -11.1748791, -1.6084337, 1.5967231
6: -16.3375626, -12.7550831, -16.3375626, -12.7550831, -2.2798905, 2.2796474
7: -4.0563126, -1.3696806, -4.0563126, -1.3696806, -2.4393096, 2.4453855
8: -6.0375504, -3.6194944, -6.0375504, -3.6194944, -2.0145197, 2.0085278
9: -11.8428965, -9.3279085, -11.8428965, -9.3279085, -1.7255714, 1.7199345

Time for backsubstitution: 21.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 6218
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 4586

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6021945, upper bound: 0.6053012
time: 9.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6021910, upper bound: 0.6087422
time: 7.10 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 37.97 seconds
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 37.97
Output dim: 2, lower bound: -0.6087668, upper bound: 0.6028562
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 37.97
Output dim: 2, lower bound: -0.6118632, upper bound: 0.5997599
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 37.97
Output dim: 2, lower bound: -0.6021945, upper bound: 0.6053012
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 37.97
Output dim: 2, lower bound: -0.6021910, upper bound: 0.6087422

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.1174488, -10.4751902, -13.1174488, -10.4751902, -1.7645597, 1.7818086
1: -7.1292858, -4.1849318, -7.1292858, -4.1849318, -2.3147640, 2.3122344
2: 9.3677406, 11.2813492, 9.3677406, 11.2813492, -1.5458884, 1.5321827
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9127784, 1.9106393
4: -9.4387360, -6.7248478, -9.4387360, -6.7248478, -1.9540968, 1.9593763
5: -13.7978468, -11.1748791, -13.7978468, -11.1748791, -1.5903883, 1.6053624
6: -16.3375626, -12.7550831, -16.3375626, -12.7550831, -2.2749257, 2.2775989
7: -4.0563126, -1.3696806, -4.0563126, -1.3696806, -2.4411049, 2.4372368
8: -6.0375504, -3.6194944, -6.0375504, -3.6194944, -2.0017776, 2.0112543
9: -11.8428965, -9.3279085, -11.8428965, -9.3279085, -1.7187579, 1.7250023

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6218
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 4586
type: DSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6218

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118631, upper bound: 0.5995927
time: 6.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6116982, upper bound: 0.5997575
time: 7.66 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 36.56 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.56
Output dim: 2, lower bound: -0.6118631, upper bound: 0.5995927
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.56
Output dim: 2, lower bound: -0.6116982, upper bound: 0.5997575

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.1174488, -10.4751902, -13.1174488, -10.4751902, -1.7625299, 1.7809584
1: -7.1292858, -4.1849318, -7.1292858, -4.1849318, -2.2952261, 2.2899022
2: 9.3677406, 11.2813492, 9.3677406, 11.2813492, -1.5447421, 1.5308180
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.8984442, 1.8942566
4: -9.4387360, -6.7248478, -9.4387360, -6.7248478, -1.9568076, 1.9624596
5: -13.7978468, -11.1748791, -13.7978468, -11.1748791, -1.5876756, 1.6015406
6: -16.3375626, -12.7550831, -16.3375626, -12.7550831, -2.2597919, 2.2643557
7: -4.0563126, -1.3696806, -4.0563126, -1.3696806, -2.4260893, 2.4241362
8: -6.0375504, -3.6194944, -6.0375504, -3.6194944, -2.0005202, 2.0103540
9: -11.8428965, -9.3279085, -11.8428965, -9.3279085, -1.7068405, 1.7145720

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 4586
type: DSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6087390, upper bound: 0.5986673
time: 4.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6052981, upper bound: 0.5986682
time: 7.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.1174488, -10.4751902, -13.1174488, -10.4751902, -1.7637095, 1.7797787
1: -7.1292858, -4.1849318, -7.1292858, -4.1849318, -2.2924318, 2.2926960
2: 9.3677406, 11.2813492, 9.3677406, 11.2813492, -1.5445232, 1.5310369
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.8963957, 1.8963046
4: -9.4387360, -6.7248478, -9.4387360, -6.7248478, -1.9571800, 1.9620876
5: -13.7978468, -11.1748791, -13.7978468, -11.1748791, -1.5865664, 1.6026497
6: -16.3375626, -12.7550831, -16.3375626, -12.7550831, -2.2616830, 2.2624648
7: -4.0563126, -1.3696806, -4.0563126, -1.3696806, -2.4280043, 2.4222212
8: -6.0375504, -3.6194944, -6.0375504, -3.6194944, -2.0008774, 2.0099969
9: -11.8428965, -9.3279085, -11.8428965, -9.3279085, -1.7083273, 1.7130849

Time for backsubstitution: 22.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 4586

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6085741, upper bound: 0.5988321
time: 5.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6051332, upper bound: 0.5988331
time: 8.90 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 36.34 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 36.34
Output dim: 2, lower bound: -0.6087390, upper bound: 0.5986673
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 36.34
Output dim: 2, lower bound: -0.6052981, upper bound: 0.5986682
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 36.34
Output dim: 2, lower bound: -0.6085741, upper bound: 0.5988321
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 36.34
Output dim: 2, lower bound: -0.6051332, upper bound: 0.5988331

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 61.86 + 276.46 = 338.32 seconds
