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
execution time: IAR + RelationalAnalysis = 23.64 + 35.09 = 58.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.5783445, upper bound: 0.5783522

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5760
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 4642

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 5760

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5723301, upper bound: 0.5783469
time: 4.14 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5723301, upper bound: 0.5723379
time: 4.63 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.07 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.07
Output dim: 1, lower bound: -0.5723301, upper bound: 0.5783469
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 9.07
Output dim: 1, lower bound: -0.5723301, upper bound: 0.5723379

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1780944, 1.1727962
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.1402130, 1.1451998
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9720068, 0.9643576
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -1.0888371, 1.0773633
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.9514432, 0.9506311
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -1.0102630, 1.0102100
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.9861946, 0.9777923
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.8114512, 0.8051648
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2981477, 1.2972136
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7568979, 0.7433326

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 4642

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 5844

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5677734, upper bound: 0.5783473
time: 3.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5677734, upper bound: 0.5737901
time: 4.23 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.83 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.83
Output dim: 1, lower bound: -0.5677734, upper bound: 0.5783473
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 29.83
Output dim: 1, lower bound: -0.5677734, upper bound: 0.5737901

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1303339, 1.1090636
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.1300559, 1.1375818
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9393101, 0.9207637
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -1.0658522, 1.0601258
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.9269996, 0.9323053
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9949245, 0.9867916
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.9574108, 0.9394116
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.7966323, 0.7940538
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2597475, 1.2459807
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7404528, 0.7310045

Time for backsubstitution: 21.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 4642

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 5746

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5593206, upper bound: 0.5783402
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5677758, upper bound: 0.5698729
time: 4.32 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.91 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.91
Output dim: 1, lower bound: -0.5593206, upper bound: 0.5783402
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 29.91
Output dim: 1, lower bound: -0.5677758, upper bound: 0.5698729

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1152058, 1.0888970
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.1040325, 1.1231284
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9291706, 0.9031906
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -1.0051672, 0.9792037
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.9102793, 0.9197707
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9653797, 0.9646344
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.8759365, 0.8783319
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.7383206, 0.7162907
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2425208, 1.2230148
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7286766, 0.7153113

Time for backsubstitution: 22.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 4642

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 930

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5587594, upper bound: 0.5783344
time: 4.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5587532, upper bound: 0.5732048
time: 4.33 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.43 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 1, lower bound: -0.5587594, upper bound: 0.5783344
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.43
Output dim: 1, lower bound: -0.5587532, upper bound: 0.5732048

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1182756, 1.0906157
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.1013727, 1.1211319
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9241943, 0.8965626
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -0.9975429, 0.9734852
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.8994918, 0.9116745
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9654703, 0.9646969
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.8733673, 0.8749075
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.7374499, 0.7158260
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2380733, 1.2170858
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7224371, 0.7106366

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 4642

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 5830

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5502538, upper bound: 0.5783358
time: 3.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5587602, upper bound: 0.5697569
time: 4.36 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 30.88 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.88
Output dim: 1, lower bound: -0.5502538, upper bound: 0.5783358
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.88
Output dim: 1, lower bound: -0.5587602, upper bound: 0.5697569

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1095138, 1.0755472
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.0961299, 1.1171942
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9129190, 0.8815436
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -0.9914045, 0.9688809
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.8789668, 0.8962741
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9652419, 0.9645262
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.8746011, 0.8758657
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.7383442, 0.7165399
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2321815, 1.2092419
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7049236, 0.6975002

Time for backsubstitution: 22.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 4642

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 5816

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5411844, upper bound: 0.5783350
time: 4.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5411846, upper bound: 0.5694220
time: 4.24 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 31.15 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 31.15
Output dim: 1, lower bound: -0.5411844, upper bound: 0.5783350
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 31.15
Output dim: 1, lower bound: -0.5411846, upper bound: 0.5694220

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1283827, 1.0906060
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.0901294, 1.1126895
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9217381, 0.8925958
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -0.9942279, 0.9689338
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.8686628, 0.8885412
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9432235, 0.9480133
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.8770769, 0.8793442
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.7127416, 0.6823945
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2333207, 1.2106695
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7135184, 0.7043617

Time for backsubstitution: 22.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 4642

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 846

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5400651, upper bound: 0.5778955
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5400647, upper bound: 0.5712584
time: 3.86 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 30.38 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 30.38
Output dim: 1, lower bound: -0.5400651, upper bound: 0.5778955
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 30.38
Output dim: 1, lower bound: -0.5400647, upper bound: 0.5712584

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1295300, 1.0914016
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.0839634, 1.1089191
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9309466, 0.8989844
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -0.9937325, 0.9685624
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.8675632, 0.8877163
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9450970, 0.9523330
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.8820276, 0.8826022
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.7112648, 0.6789761
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2403374, 1.2207956
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7148833, 0.7063297

Time for backsubstitution: 22.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 4642

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 902

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5400010, upper bound: 0.5778887
time: 4.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5400009, upper bound: 0.5687671
time: 4.12 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 31.02 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 31.02
Output dim: 1, lower bound: -0.5400010, upper bound: 0.5778887
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 31.02
Output dim: 1, lower bound: -0.5400009, upper bound: 0.5687671

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1268473, 1.0893898
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.0852571, 1.1098185
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9329109, 0.9018111
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -0.9957209, 0.9699454
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.8677058, 0.8878160
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9380064, 0.9488516
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.8837929, 0.8851442
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.7065957, 0.6718636
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2405024, 1.2210336
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7152960, 0.7066164

Time for backsubstitution: 22.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 4642

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 916

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5396587, upper bound: 0.5778813
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5396582, upper bound: 0.5689612
time: 4.31 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 30.81 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 30.81
Output dim: 1, lower bound: -0.5396587, upper bound: 0.5778813
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 30.81
Output dim: 1, lower bound: -0.5396582, upper bound: 0.5689612

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1336012, 1.0940773
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.0828471, 1.1080084
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9294453, 0.8971896
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -0.9938495, 0.9685435
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.8612967, 0.8830023
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9381924, 0.9489837
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.8833871, 0.8848395
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.7066665, 0.6719127
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2386918, 1.2186222
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7112346, 0.7035682

Time for backsubstitution: 22.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 4642

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 832

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5381057, upper bound: 0.5772876
time: 4.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5381061, upper bound: 0.5678845
time: 4.17 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 30.86 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 30.86
Output dim: 1, lower bound: -0.5381057, upper bound: 0.5772876
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 30.86
Output dim: 1, lower bound: -0.5381061, upper bound: 0.5678845

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1310654, 1.0904791
1: 1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.0897694, 1.1179819
2: -6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9351654, 0.9011598
3: -10.5567799, -8.9935284, -10.5567799, -8.9935284, -0.9859037, 0.9556208
4: -4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.8530397, 0.8768115
5: -8.6873465, -7.4558282, -8.6873465, -7.4558282, -0.9301229, 0.9429231
6: -8.2650509, -6.7415175, -8.2650509, -6.7415175, -0.8623483, 0.8690641
7: -7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.6901956, 0.6499679
8: -0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.2334838, 1.2116766
9: -5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7031405, 0.6927809

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4642

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 4642

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5326376, upper bound: 0.5772800
time: 4.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5380976, upper bound: 0.5718569
time: 4.37 seconds

## Summary of splitting (split count: 10)
- Time for DS candidates: 31.08 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 11, time: 31.08
Output dim: 1, lower bound: -0.5326376, upper bound: 0.5772800
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 31.08
Output dim: 1, lower bound: -0.5380976, upper bound: 0.5718569

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

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 752
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 662
type: DSZ, layer: 3, pos: 2222
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1949
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 74
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 2243
type: DSZ, layer: 3, pos: 2389
type: DSZ, layer: 3, pos: 430
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1221
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 2382

Time for candidate selection: 0.46 seconds

### Candidate
type: DSZ, layer: 3, pos: 752

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5158221, upper bound: 0.5547566
time: 4.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5143101, upper bound: 0.5578584
time: 4.24 seconds

## Summary of splitting (split count: 11)
- Time for DS candidates: 30.98 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 12, time: 30.98
Output dim: 1, lower bound: -0.5158221, upper bound: 0.5547566
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 12, time: 30.98
Output dim: 1, lower bound: -0.5143101, upper bound: 0.5578584

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 58.74 + 347.39 = 406.13 seconds
