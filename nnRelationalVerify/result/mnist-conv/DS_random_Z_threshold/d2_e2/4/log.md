## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.3662410892


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5508573, 0.5508573)
1: (-14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8362961, 0.8362958)
2: (-7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6987977, 0.6987977)
3: (-8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.7168481, 0.7168481)
4: (-12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7461720, 0.7461717)
5: (-5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6630130, 0.6630135)
6: (-3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6843560, 0.6843560)
7: (-8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6726513, 0.6726513)
8: (-3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.6035771, 0.6035774)
9: (-2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6827922, 0.6827924)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.74 + 36.14 = 59.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.3684516, upper bound: 0.3684522

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6126
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6126

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3684502, upper bound: 0.3675058
time: 5.26 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3675058, upper bound: 0.3684503
time: 4.92 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.19 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.19
Output dim: 0, lower bound: -0.3684502, upper bound: 0.3675058
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.19
Output dim: 0, lower bound: -0.3675058, upper bound: 0.3684503

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5481918, 0.5468607
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8242493, 0.8187785
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6830788, 0.6870065
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.7150035, 0.7138703
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7448535, 0.7440729
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6483936, 0.6435030
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6724184, 0.6684527
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6533794, 0.6581984
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5940957, 0.5909276
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6637068, 0.6556916

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 554

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3668444, upper bound: 0.3675036
time: 4.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3684475, upper bound: 0.3659001
time: 3.34 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5468607, 0.5481918
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8187785, 0.8242493
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6870065, 0.6830788
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.7138703, 0.7150035
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7440729, 0.7448535
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6435032, 0.6483939
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6684525, 0.6724184
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6581984, 0.6533794
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5909276, 0.5940957
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6556916, 0.6637073

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4559

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3675011, upper bound: 0.3668057
time: 3.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3658611, upper bound: 0.3684459
time: 5.14 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.22 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.22
Output dim: 0, lower bound: -0.3668444, upper bound: 0.3675036
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.22
Output dim: 0, lower bound: -0.3684475, upper bound: 0.3659001
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.22
Output dim: 0, lower bound: -0.3675011, upper bound: 0.3668057
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.22
Output dim: 0, lower bound: -0.3658611, upper bound: 0.3684459

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5421400, 0.5427885
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8034277, 0.8043077
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6464555, 0.6597667
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6943989, 0.6863880
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7209530, 0.7271652
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6359816, 0.6263237
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6580777, 0.6464305
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6469860, 0.6553328
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5840063, 0.5760581
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6654639, 0.6581972

Time for backsubstitution: 22.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 4559

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3662810, upper bound: 0.3675025
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3668433, upper bound: 0.3669360
time: 5.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5441196, 0.5408089
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8097782, 0.7979567
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6558392, 0.6503830
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6875212, 0.6932657
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7279453, 0.7201724
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6312151, 0.6310909
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6503963, 0.6541121
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6505136, 0.6518052
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5792265, 0.5808380
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6662130, 0.6574485

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 4559

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6127

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3684458, upper bound: 0.3641044
time: 3.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3666517, upper bound: 0.3658979
time: 4.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5423932, 0.5422440
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8168874, 0.8212171
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6775427, 0.6759830
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.7173407, 0.7174759
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7395124, 0.7387748
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6384149, 0.6416161
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6560163, 0.6558073
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6457939, 0.6441803
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5801728, 0.5797479
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6355281, 0.6360426

Time for backsubstitution: 23.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6127

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3674993, upper bound: 0.3650084
time: 5.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3657061, upper bound: 0.3668040
time: 5.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5409131, 0.5437248
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8157463, 0.8223567
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6799088, 0.6736145
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.7163427, 0.7184734
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7379942, 0.7402921
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6367249, 0.6433067
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6518416, 0.6599791
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6489968, 0.6409750
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5765793, 0.5833375
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6280270, 0.6435356

Time for backsubstitution: 22.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3652939, upper bound: 0.3684444
time: 4.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3658599, upper bound: 0.3678708
time: 4.21 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.70 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.3662810, upper bound: 0.3675025
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.3668433, upper bound: 0.3669360
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.3684458, upper bound: 0.3641044
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.3666517, upper bound: 0.3658979
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.3674993, upper bound: 0.3650084
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.3657061, upper bound: 0.3668040
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.3652939, upper bound: 0.3684444
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.3658599, upper bound: 0.3678708

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5409310, 0.5418828
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.7910461, 0.7866178
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6331592, 0.6497946
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6904299, 0.6797020
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7166324, 0.7213950
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6280956, 0.6158195
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6467366, 0.6313043
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6338849, 0.6455002
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5771456, 0.5667224
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6552787, 0.6432514

Time for backsubstitution: 22.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4559

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3662763, upper bound: 0.3658574
time: 4.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3646361, upper bound: 0.3674977
time: 4.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5412340, 0.5415795
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.7857375, 0.7919364
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6364889, 0.6464703
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6877129, 0.6824224
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7151833, 0.7228465
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6254778, 0.6184440
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6429515, 0.6350954
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6371646, 0.6422315
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5746708, 0.5692043
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6505179, 0.6480236

Time for backsubstitution: 22.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 4559

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3668416, upper bound: 0.3651409
time: 4.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3650474, upper bound: 0.3669344
time: 5.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5389922, 0.5339680
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8057513, 0.7923565
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6458366, 0.6430390
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6898313, 0.6938107
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7247000, 0.7182851
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6310520, 0.6319883
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6331882, 0.6311741
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6426730, 0.6459253
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5699842, 0.5685122
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6536193, 0.6391568

Time for backsubstitution: 22.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 4559

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3678707, upper bound: 0.3641032
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3684447, upper bound: 0.3635419
time: 7.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5372784, 0.5356815
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8041778, 0.7939296
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6484950, 0.6403806
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6880665, 0.6955748
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7260575, 0.7169266
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6321115, 0.6309280
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6274581, 0.6369030
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6446323, 0.6439645
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5669010, 0.5715950
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6479211, 0.6448541

Time for backsubstitution: 23.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 4559

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3660826, upper bound: 0.3658965
time: 5.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3666506, upper bound: 0.3653353
time: 5.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5372632, 0.5354016
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8128567, 0.8156152
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6675262, 0.6686192
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.7196445, 0.7180181
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7362728, 0.7368932
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6382599, 0.6425204
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6387856, 0.6328583
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6379495, 0.6382906
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5709133, 0.5674155
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6228795, 0.6177127

Time for backsubstitution: 23.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 554

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3669324, upper bound: 0.3650079
time: 5.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3674981, upper bound: 0.3644417
time: 4.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5355506, 0.5371151
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8112850, 0.8171883
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6701841, 0.6659665
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.7178826, 0.7197828
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7376318, 0.7355356
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6393194, 0.6414607
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6330674, 0.6385882
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6399102, 0.6363356
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5678401, 0.5704987
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6171985, 0.6234109

Time for backsubstitution: 22.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 554

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3640997, upper bound: 0.3668012
time: 5.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3657034, upper bound: 0.3651987
time: 4.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5397034, 0.5428181
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8033729, 0.8046646
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6666129, 0.6636486
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.7123754, 0.7117860
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7336755, 0.7345219
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6288443, 0.6328011
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6405106, 0.6448572
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6358938, 0.6311522
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5697255, 0.5740016
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6178508, 0.6285872

Time for backsubstitution: 22.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 554

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3636920, upper bound: 0.3684416
time: 4.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6127

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3652926, upper bound: 0.3666490
time: 8.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3634973, upper bound: 0.3684427
time: 5.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5400066, 0.5425150
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.7980542, 0.8099742
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6699388, 0.6603189
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.7096550, 0.7145030
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7322245, 0.7359724
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6262197, 0.6354201
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6367197, 0.6486428
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6391640, 0.6278725
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5672436, 0.5764778
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6130786, 0.6333499

Time for backsubstitution: 22.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6127
type: DSZ, layer: 1, pos: 554

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6127

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3658586, upper bound: 0.3660812
time: 4.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3640633, upper bound: 0.3678686
time: 4.64 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 32.15 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3662763, upper bound: 0.3658574
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3646361, upper bound: 0.3674977
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3668416, upper bound: 0.3651409
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3650474, upper bound: 0.3669344
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3678707, upper bound: 0.3641032
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3684447, upper bound: 0.3635419
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3660826, upper bound: 0.3658965
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3666506, upper bound: 0.3653353
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3669324, upper bound: 0.3650079
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3674981, upper bound: 0.3644417
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3640997, upper bound: 0.3668012
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3657034, upper bound: 0.3651987
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3652926, upper bound: 0.3666490
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3634973, upper bound: 0.3684427
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3658586, upper bound: 0.3660812
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.15
Output dim: 0, lower bound: -0.3640633, upper bound: 0.3678686

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5364635, 0.5359344
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.7891531, 0.7835834
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6236954, 0.6426990
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6938980, 0.6821725
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7120709, 0.7153156
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6230097, 0.6090415
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6343019, 0.6146972
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6214795, 0.6362994
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5663884, 0.5523746
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6351075, 0.6155853

Time for backsubstitution: 23.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3662745, upper bound: 0.3640612
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3644811, upper bound: 0.3658560
time: 5.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5349829, 0.5374143
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.7880125, 0.7847245
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6260638, 0.6403310
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6929004, 0.6831703
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7105527, 0.7168338
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6213179, 0.6107314
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6301296, 0.6188719
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6246848, 0.6330950
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5627978, 0.5559676
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6276126, 0.6230860

Time for backsubstitution: 23.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3646349, upper bound: 0.3657021
time: 8.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3628394, upper bound: 0.3674955
time: 4.88 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 36.58 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.58
Output dim: 0, lower bound: -0.3662745, upper bound: 0.3640612
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 36.58
Output dim: 0, lower bound: -0.3644811, upper bound: 0.3658560
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 36.58
Output dim: 0, lower bound: -0.3646349, upper bound: 0.3657021
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.58
Output dim: 0, lower bound: -0.3628394, upper bound: 0.3674955
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.58
Output dim: 0, lower bound: -0.3668416, upper bound: 0.3651409
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.58
Output dim: 0, lower bound: -0.3650474, upper bound: 0.3669344
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.58
Output dim: 0, lower bound: -0.3678707, upper bound: 0.3641032
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.58
Output dim: 0, lower bound: -0.3684447, upper bound: 0.3635419
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.58
Output dim: 0, lower bound: -0.3666506, upper bound: 0.3653353
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.58
Output dim: 0, lower bound: -0.3669324, upper bound: 0.3650079
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.58
Output dim: 0, lower bound: -0.3674981, upper bound: 0.3644417
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.58
Output dim: 0, lower bound: -0.3640997, upper bound: 0.3668012
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.58
Output dim: 0, lower bound: -0.3652926, upper bound: 0.3666490
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.58
Output dim: 0, lower bound: -0.3634973, upper bound: 0.3684427
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.58
Output dim: 0, lower bound: -0.3640633, upper bound: 0.3678686

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 59.89 + 549.02 = 608.91 seconds
