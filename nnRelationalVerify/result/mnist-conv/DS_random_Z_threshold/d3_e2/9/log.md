## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.5754619315


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1536942, 1.1536946)
1: (1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.1601706, 1.1601706)
2: (-6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9770384, 0.9770386)
3: (-10.5567799, -8.9935284, -10.5567799, -8.9935284, -1.1204410, 1.1204410)
4: (-4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.9474130, 0.9474130)
5: (-8.6873465, -7.4558282, -8.6873465, -7.4558282, -1.0102134, 1.0102134)
6: (-8.2650509, -6.7415175, -8.2650509, -6.7415175, -1.0024719, 1.0024717)
7: (-7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.8303411, 0.8303411)
8: (-0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.3009887, 1.3009882)
9: (-5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7976398, 0.7976398)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.99 + 34.46 = 58.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.5783445, upper bound: 0.5783522

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4642
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 5760
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 5844

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4642

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5728966, upper bound: 0.5783455
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783364, upper bound: 0.5729060
time: 3.80 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.41 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.41
Output dim: 1, lower bound: -0.5728966, upper bound: 0.5783455
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.41
Output dim: 1, lower bound: -0.5783364, upper bound: 0.5729060

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1536484, 1.1532297
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.1556516, 1.1597347
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9769497, 0.9761260
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -1.1198547, 1.1144018
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.9403358, 0.9467235
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -1.0083642, 1.0100341
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -1.0016747, 0.9943037
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.8228681, 0.8296103
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2999654, 1.2904124
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7897298, 0.7968650

Time for backsubstitution: 21.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 5760
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 916

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5639687, upper bound: 0.5783325
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5639688, upper bound: 0.5694169
time: 4.41 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1532297, 1.1536484
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.1597347, 1.1556516
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9761257, 0.9769497
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -1.1144016, 1.1198549
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.9467235, 0.9403358
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -1.0100341, 1.0083642
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.9943037, 1.0016749
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.8296103, 0.8228683
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2904124, 1.2999659
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7968647, 0.7897301

Time for backsubstitution: 21.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5760
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5760

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5723220, upper bound: 0.5728992
time: 4.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5723220, upper bound: 0.5668902
time: 4.43 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.38 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.38
Output dim: 1, lower bound: -0.5639687, upper bound: 0.5783325
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 30.38
Output dim: 1, lower bound: -0.5639688, upper bound: 0.5694169
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 30.38
Output dim: 1, lower bound: -0.5723220, upper bound: 0.5728992
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 30.38
Output dim: 1, lower bound: -0.5723220, upper bound: 0.5668902

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1474380, 1.1449625
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.1532416, 1.1579204
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9734759, 0.9715056
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -1.1179824, 1.1129987
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.9339271, 0.9419036
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -1.0085497, 1.0101657
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -1.0019045, 0.9946349
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.8229392, 0.8296595
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2981448, 1.2880006
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7856700, 0.7938185

Time for backsubstitution: 23.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 5760
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 902

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5548726, upper bound: 0.5783302
time: 5.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5548725, upper bound: 0.5692582
time: 4.02 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 32.48 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.48
Output dim: 1, lower bound: -0.5548726, upper bound: 0.5783302
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 32.48
Output dim: 1, lower bound: -0.5548725, upper bound: 0.5692582

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1460404, 1.1442363
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.1545353, 1.1588197
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9754400, 0.9743323
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -1.1161597, 1.1105704
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.9340701, 0.9420033
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9941325, 0.9993539
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -1.0021553, 0.9956546
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.8155932, 0.8198738
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2983108, 1.2882390
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7860777, 0.7941051

Time for backsubstitution: 23.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 5760
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 846

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5830

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5548705, upper bound: 0.5783306
time: 4.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5548841, upper bound: 0.5693965
time: 4.13 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.60 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.60
Output dim: 1, lower bound: -0.5548705, upper bound: 0.5783306
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.60
Output dim: 1, lower bound: -0.5548841, upper bound: 0.5693965

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1502495, 1.1421239
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.1492925, 1.1548872
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9641771, 0.9593134
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -1.1100254, 1.1059692
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.9135456, 0.9266143
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9939022, 0.9991803
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -1.0027552, 0.9959784
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.8164878, 0.8205876
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2924323, 1.2803979
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7685666, 0.7809746

Time for backsubstitution: 22.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 5760

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5746

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5465078, upper bound: 0.5783253
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5465059, upper bound: 0.5698569
time: 4.53 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 31.10 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 31.10
Output dim: 1, lower bound: -0.5465078, upper bound: 0.5783253
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 31.10
Output dim: 1, lower bound: -0.5465059, upper bound: 0.5698569

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1351223, 1.1219559
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.1232738, 1.1404419
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9540501, 0.9417479
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -1.0493650, 1.0250676
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.8968253, 0.9140797
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9643588, 0.9770241
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.9212787, 0.9348960
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.7581840, 0.7428310
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2752070, 1.2574329
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7568026, 0.7652893

Time for backsubstitution: 22.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5760
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5760

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5404756, upper bound: 0.5783125
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5465039, upper bound: 0.5723102
time: 3.91 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 30.69 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 30.69
Output dim: 1, lower bound: -0.5404756, upper bound: 0.5783125
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 30.69
Output dim: 1, lower bound: -0.5465039, upper bound: 0.5723102

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1595216, 1.1410599
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.1033125, 1.1254640
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9490056, 0.9290593
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -1.0177364, 0.9819684
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.9008551, 0.9172978
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9644089, 0.9770212
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.9050035, 0.9102180
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.7392852, 0.7176476
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2723651, 1.2536573
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7160487, 0.7109745

Time for backsubstitution: 22.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 5844

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 930

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5353662, upper bound: 0.5783075
time: 3.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5353661, upper bound: 0.5731942
time: 4.63 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 31.21 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 31.21
Output dim: 1, lower bound: -0.5353662, upper bound: 0.5783075
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 31.21
Output dim: 1, lower bound: -0.5353661, upper bound: 0.5731942

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1554556, 1.1356497
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.1006532, 1.1234646
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9440181, 0.9224339
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -1.0101123, 0.9762509
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.8900714, 0.9092045
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9643145, 0.9769106
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.9024248, 0.9067926
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.7376654, 0.7164333
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2679186, 1.2477303
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7098088, 0.7062976

Time for backsubstitution: 22.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5816

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5353424, upper bound: 0.5783127
time: 4.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5353690, upper bound: 0.5692194
time: 4.31 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 31.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 31.13
Output dim: 1, lower bound: -0.5353424, upper bound: 0.5783127
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 31.13
Output dim: 1, lower bound: -0.5353690, upper bound: 0.5692194

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1730390, 1.1494250
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.0946527, 1.1189590
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9528372, 0.9334860
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -1.0167446, 0.9801133
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.8797703, 0.9014740
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9496193, 0.9677253
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.9064116, 0.9117899
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.7147386, 0.6849606
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2690578, 1.2491584
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7184067, 0.7131573

Time for backsubstitution: 23.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 846

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5844

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5353045, upper bound: 0.5783143
time: 4.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5353194, upper bound: 0.5731678
time: 4.62 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 32.33 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 32.33
Output dim: 1, lower bound: -0.5353045, upper bound: 0.5783143
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 32.33
Output dim: 1, lower bound: -0.5353194, upper bound: 0.5731678

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1324062, 1.0928149
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.0844951, 1.1113434
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9201491, 0.8898897
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -0.9937587, 0.9628751
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.8553190, 0.8831372
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9344692, 0.9444838
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.8776395, 0.8734143
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.7006705, 0.6746004
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2306528, 1.1979203
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7019598, 0.7008251

Time for backsubstitution: 23.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 846

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 832

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5337517, upper bound: 0.5777141
time: 4.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5337517, upper bound: 0.5682937
time: 4.65 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 31.81 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 31.81
Output dim: 1, lower bound: -0.5337517, upper bound: 0.5777141
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 31.81
Output dim: 1, lower bound: -0.5337517, upper bound: 0.5682937

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1298718, 1.0892181
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.0914197, 1.1213193
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9258733, 0.8938632
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -0.9858131, 0.9499526
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.8470612, 0.8769450
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9264030, 0.9384270
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.8566003, 0.8576379
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.6842034, 0.6526594
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2254429, 1.1909733
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.6938658, 0.6900378

Time for backsubstitution: 23.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 846

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 846

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5326376, upper bound: 0.5772800
time: 4.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5326371, upper bound: 0.5706594
time: 4.00 seconds

## Summary of splitting (split count: 10)
- Time for DS candidates: 31.93 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 11, time: 31.93
Output dim: 1, lower bound: -0.5326376, upper bound: 0.5772800
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 31.93
Output dim: 1, lower bound: -0.5326371, upper bound: 0.5706594

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1310186, 1.0900137
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.0852499, 1.1175456
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9350770, 0.9002476
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -0.9853175, 0.9495816
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.8459625, 0.8761210
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9282737, 0.9427433
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.8615513, 0.8608959
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.6827228, 0.6492372
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2324605, 1.2011003
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.6952312, 0.6920063

Time for backsubstitution: 23.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 662
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 2222
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 74
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1221
type: DSZ, layer: 3, pos: 752
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2243
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 1949
type: DSZ, layer: 3, pos: 921

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 724

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5197591, upper bound: 0.5695130
time: 4.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5250963, upper bound: 0.5647308
time: 4.15 seconds

## Summary of splitting (split count: 11)
- Time for DS candidates: 31.54 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 12, time: 31.54
Output dim: 1, lower bound: -0.5197591, upper bound: 0.5695130
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 12, time: 31.54
Output dim: 1, lower bound: -0.5250963, upper bound: 0.5647308

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 58.45 + 383.57 = 442.02 seconds
