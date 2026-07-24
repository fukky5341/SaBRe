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
execution time: IAR + RelationalAnalysis = 22.34 + 37.48 = 59.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.6118839, upper bound: 0.6118833

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 6198
type: DSZ, layer: 1, pos: 4586
type: DSZ, layer: 1, pos: 6218

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5735

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118729, upper bound: 0.6062237
time: 9.03 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6062216, upper bound: 0.6118723
time: 11.42 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 20.54 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 20.54
Output dim: 2, lower bound: -0.6118729, upper bound: 0.6062237
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 20.54
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

Time for backsubstitution: 21.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 6198
type: DSZ, layer: 1, pos: 4586
type: DSZ, layer: 1, pos: 6218

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5762

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6085120, upper bound: 0.6062178
time: 8.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118697, upper bound: 0.6028628
time: 4.76 seconds

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

Time for backsubstitution: 21.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 6198
type: DSZ, layer: 1, pos: 4586
type: DSZ, layer: 1, pos: 6218

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5762

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6028606, upper bound: 0.6118693
time: 9.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6062183, upper bound: 0.6085117
time: 8.81 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 39.88 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 39.88
Output dim: 2, lower bound: -0.6085120, upper bound: 0.6062178
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 39.88
Output dim: 2, lower bound: -0.6118697, upper bound: 0.6028628
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 39.88
Output dim: 2, lower bound: -0.6028606, upper bound: 0.6118693
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 39.88
Output dim: 2, lower bound: -0.6062183, upper bound: 0.6085117

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

Time for backsubstitution: 21.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 6198
type: DSZ, layer: 1, pos: 4586
type: DSZ, layer: 1, pos: 6218

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 457

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6113804, upper bound: 0.6028340
time: 11.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118406, upper bound: 0.6023701
time: 8.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.1174488, -10.4751902, -13.1174488, -10.4751902, -1.7825794, 1.7651200
1: -7.1292858, -4.1849318, -7.1292858, -4.1849318, -2.3130741, 2.3153729
2: 9.3677406, 11.2813492, 9.3677406, 11.2813492, -1.5448170, 1.5550528
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9127545, 1.9143119
4: -9.4387360, -6.7248478, -9.4387360, -6.7248478, -1.9602332, 1.9547186
5: -13.7978468, -11.1748791, -13.7978468, -11.1748791, -1.6194291, 1.6097789
6: -16.3375626, -12.7550831, -16.3375626, -12.7550831, -2.2814984, 2.2802999
7: -4.0563126, -1.3696806, -4.0563126, -1.3696806, -2.4606962, 2.4581246
8: -6.0375504, -3.6194944, -6.0375504, -3.6194944, -2.0238423, 2.0191255
9: -11.8428965, -9.3279085, -11.8428965, -9.3279085, -1.7305424, 1.7227783

Time for backsubstitution: 22.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 6198
type: DSZ, layer: 1, pos: 4586
type: DSZ, layer: 1, pos: 6218

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 457

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6023710, upper bound: 0.6118400
time: 5.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6028318, upper bound: 0.6113825
time: 6.75 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 35.15 seconds
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 35.15
Output dim: 2, lower bound: -0.6113804, upper bound: 0.6028340
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 35.15
Output dim: 2, lower bound: -0.6118406, upper bound: 0.6023701
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 35.15
Output dim: 2, lower bound: -0.6023710, upper bound: 0.6118400
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 35.15
Output dim: 2, lower bound: -0.6028318, upper bound: 0.6113825

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.1174488, -10.4751902, -13.1174488, -10.4751902, -1.7647886, 1.7841446
1: -7.1292858, -4.1849318, -7.1292858, -4.1849318, -2.3159151, 2.3129587
2: 9.3677406, 11.2813492, 9.3677406, 11.2813492, -1.5547681, 1.5461588
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9146285, 1.9126863
4: -9.4387360, -6.7248478, -9.4387360, -6.7248478, -1.9591012, 1.9593220
5: -13.7978468, -11.1748791, -13.7978468, -11.1748791, -1.6093678, 1.6213815
6: -16.3375626, -12.7550831, -16.3375626, -12.7550831, -2.2820077, 2.2811415
7: -4.0563126, -1.3696806, -4.0563126, -1.3696806, -2.4603848, 2.4602165
8: -6.0375504, -3.6194944, -6.0375504, -3.6194944, -2.0195308, 2.0237589
9: -11.8428965, -9.3279085, -11.8428965, -9.3279085, -1.7229526, 1.7305052

Time for backsubstitution: 22.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 6198
type: DSZ, layer: 1, pos: 4586
type: DSZ, layer: 1, pos: 6218

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6082565, upper bound: 0.6019046
time: 8.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6048154, upper bound: 0.6019097
time: 5.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.1174488, -10.4751902, -13.1174488, -10.4751902, -1.7651200, 1.7822483
1: -7.1292858, -4.1849318, -7.1292858, -4.1849318, -2.3152590, 2.3130736
2: 9.3677406, 11.2813492, 9.3677406, 11.2813492, -1.5550528, 1.5445323
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9142442, 1.9127545
4: -9.4387360, -6.7248478, -9.4387360, -6.7248478, -1.9538074, 1.9602332
5: -13.7978468, -11.1748791, -13.7978468, -11.1748791, -1.6097789, 1.6190181
6: -16.3375626, -12.7550831, -16.3375626, -12.7550831, -2.2799430, 2.2814982
7: -4.0563126, -1.3696806, -4.0563126, -1.3696806, -2.4576459, 2.4606957
8: -6.0375504, -3.6194944, -6.0375504, -3.6194944, -2.0190415, 2.0238423
9: -11.8428965, -9.3279085, -11.8428965, -9.3279085, -1.7227414, 1.7305422

Time for backsubstitution: 22.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 6198
type: DSZ, layer: 1, pos: 4586
type: DSZ, layer: 1, pos: 6218

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6087150, upper bound: 0.6014434
time: 8.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6052757, upper bound: 0.6014463
time: 9.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.1174488, -10.4751902, -13.1174488, -10.4751902, -1.7822480, 1.7666836
1: -7.1292858, -4.1849318, -7.1292858, -4.1849318, -2.3136148, 2.3152580
2: 9.3677406, 11.2813492, 9.3677406, 11.2813492, -1.5445323, 1.5563931
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9130702, 1.9142442
4: -9.4387360, -6.7248478, -9.4387360, -6.7248478, -1.9646158, 1.9538074
5: -13.7978468, -11.1748791, -13.7978468, -11.1748791, -1.6190181, 1.6117301
6: -16.3375626, -12.7550831, -16.3375626, -12.7550831, -2.2832046, 2.2799432
7: -4.0563126, -1.3696806, -4.0563126, -1.3696806, -2.4629560, 2.4576454
8: -6.0375504, -3.6194944, -6.0375504, -3.6194944, -2.0242476, 2.0190415
9: -11.8428965, -9.3279085, -11.8428965, -9.3279085, -1.7307155, 1.7227416

Time for backsubstitution: 22.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 6198
type: DSZ, layer: 1, pos: 4586
type: DSZ, layer: 1, pos: 6218

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6014467, upper bound: 0.6052779
time: 4.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6014434, upper bound: 0.6087151
time: 4.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.1174488, -10.4751902, -13.1174488, -10.4751902, -1.7825794, 1.7647886
1: -7.1292858, -4.1849318, -7.1292858, -4.1849318, -2.3129587, 2.3153729
2: 9.3677406, 11.2813492, 9.3677406, 11.2813492, -1.5448170, 1.5547681
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9126868, 1.9143119
4: -9.4387360, -6.7248478, -9.4387360, -6.7248478, -1.9593225, 1.9547186
5: -13.7978468, -11.1748791, -13.7978468, -11.1748791, -1.6194291, 1.6093681
6: -16.3375626, -12.7550831, -16.3375626, -12.7550831, -2.2811413, 2.2802999
7: -4.0563126, -1.3696806, -4.0563126, -1.3696806, -2.4602170, 2.4581246
8: -6.0375504, -3.6194944, -6.0375504, -3.6194944, -2.0237589, 2.0191255
9: -11.8428965, -9.3279085, -11.8428965, -9.3279085, -1.7305052, 1.7227783

Time for backsubstitution: 22.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 6198
type: DSZ, layer: 1, pos: 4586
type: DSZ, layer: 1, pos: 6218

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6019076, upper bound: 0.6048151
time: 7.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6019042, upper bound: 0.6082558
time: 7.94 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 38.31 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 38.31
Output dim: 2, lower bound: -0.6082565, upper bound: 0.6019046
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 38.31
Output dim: 2, lower bound: -0.6048154, upper bound: 0.6019097
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 38.31
Output dim: 2, lower bound: -0.6087150, upper bound: 0.6014434
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 38.31
Output dim: 2, lower bound: -0.6052757, upper bound: 0.6014463
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 38.31
Output dim: 2, lower bound: -0.6014467, upper bound: 0.6052779
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 38.31
Output dim: 2, lower bound: -0.6014434, upper bound: 0.6087151
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 38.31
Output dim: 2, lower bound: -0.6019076, upper bound: 0.6048151
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 38.31
Output dim: 2, lower bound: -0.6019042, upper bound: 0.6082558

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 59.81 + 318.68 = 378.50 seconds
