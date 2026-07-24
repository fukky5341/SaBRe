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
execution time: IAR + RelationalAnalysis = 22.27 + 36.12 = 58.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.3684516, upper bound: 0.3684522

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 6126
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 554

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3668459, upper bound: 0.3684495
time: 4.26 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3684489, upper bound: 0.3668459
time: 5.13 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.59 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.59
Output dim: 0, lower bound: -0.3668459, upper bound: 0.3684495
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.59
Output dim: 0, lower bound: -0.3684489, upper bound: 0.3668459

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5448060, 0.5467854
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8154736, 0.8218248
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6621735, 0.6715569
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6962438, 0.6893663
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7222719, 0.7292640
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6506014, 0.6458342
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6700153, 0.6623340
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6662588, 0.6697867
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5934877, 0.5887079
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6845498, 0.6852987

Time for backsubstitution: 20.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6126
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6126

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3668444, upper bound: 0.3675036
time: 4.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3658995, upper bound: 0.3684480
time: 3.72 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5467854, 0.5448060
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8218250, 0.8154738
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6715567, 0.6621733
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6893661, 0.6962438
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7292643, 0.7222717
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6458344, 0.6506011
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6623340, 0.6700156
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6697865, 0.6662591
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5887079, 0.5934877
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6852989, 0.6845498

Time for backsubstitution: 20.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6126
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 6126

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3684475, upper bound: 0.3659001
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3675031, upper bound: 0.3668444
time: 5.99 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.48 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.48
Output dim: 0, lower bound: -0.3668444, upper bound: 0.3675036
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.48
Output dim: 0, lower bound: -0.3658995, upper bound: 0.3684480
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.48
Output dim: 0, lower bound: -0.3684475, upper bound: 0.3659001
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.48
Output dim: 0, lower bound: -0.3675031, upper bound: 0.3668444

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

Time for backsubstitution: 20.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3662810, upper bound: 0.3675025
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3668433, upper bound: 0.3669360
time: 5.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5408089, 0.5441194
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.7979565, 0.8097785
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6503832, 0.6558390
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6932657, 0.6875212
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7201724, 0.7279453
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6310906, 0.6312149
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6541119, 0.6503963
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6518049, 0.6505139
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5808377, 0.5792265
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6574488, 0.6662128

Time for backsubstitution: 21.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3653368, upper bound: 0.3684465
time: 4.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3658983, upper bound: 0.3678722
time: 6.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1

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

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3678723, upper bound: 0.3658982
time: 6.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3684463, upper bound: 0.3653374
time: 3.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5427883, 0.5421400
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8043079, 0.8034275
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6597669, 0.6464558
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6863880, 0.6943989
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7271647, 0.7209530
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6263237, 0.6359820
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6464305, 0.6580777
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6553330, 0.6469862
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5760579, 0.5840063
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6581974, 0.6654642

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3669359, upper bound: 0.3668433
time: 4.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3675019, upper bound: 0.3662811
time: 4.65 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.26 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.26
Output dim: 0, lower bound: -0.3662810, upper bound: 0.3675025
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.26
Output dim: 0, lower bound: -0.3668433, upper bound: 0.3669360
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.26
Output dim: 0, lower bound: -0.3653368, upper bound: 0.3684465
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.26
Output dim: 0, lower bound: -0.3658983, upper bound: 0.3678722
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.26
Output dim: 0, lower bound: -0.3678723, upper bound: 0.3658982
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.26
Output dim: 0, lower bound: -0.3684463, upper bound: 0.3653374
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.26
Output dim: 0, lower bound: -0.3669359, upper bound: 0.3668433
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.26
Output dim: 0, lower bound: -0.3675019, upper bound: 0.3662811

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

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 4559

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3662763, upper bound: 0.3658574
time: 4.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3646361, upper bound: 0.3674977
time: 4.00 seconds

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

Time for backsubstitution: 21.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 4559

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3668385, upper bound: 0.3652913
time: 4.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3651982, upper bound: 0.3669322
time: 3.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5395999, 0.5432136
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.7855854, 0.7920885
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6370869, 0.6458726
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6893001, 0.6808352
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7158537, 0.7221756
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6232109, 0.6207104
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6427770, 0.6352701
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6387038, 0.6406922
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5739841, 0.5698907
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6472745, 0.6512666

Time for backsubstitution: 21.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 4559

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3653325, upper bound: 0.3668019
time: 4.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3636920, upper bound: 0.3684416
time: 4.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5399032, 0.5429103
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.7802668, 0.7973971
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6404109, 0.6425426
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6865797, 0.6835525
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7144027, 0.7236252
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6205864, 0.6233287
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6389856, 0.6390550
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6419725, 0.6374125
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5715022, 0.5723658
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6425023, 0.6560273

Time for backsubstitution: 21.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 4559

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3658936, upper bound: 0.3662354
time: 4.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3642536, upper bound: 0.3678677
time: 8.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5429106, 0.5399032
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.7973976, 0.7802668
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6425428, 0.6404109
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6835525, 0.6865797
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7236247, 0.7144027
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6233282, 0.6205864
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6390548, 0.6389859
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6374125, 0.6419725
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5723658, 0.5715022
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6560273, 0.6425023

Time for backsubstitution: 21.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 4559

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3678676, upper bound: 0.3642536
time: 6.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3662351, upper bound: 0.3658937
time: 3.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5432136, 0.5395999
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.7920885, 0.7855854
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6458726, 0.6370869
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6808352, 0.6893001
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7221756, 0.7158537
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6207104, 0.6232109
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6352701, 0.6427767
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6406922, 0.6387038
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5698910, 0.5739841
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6512666, 0.6472750

Time for backsubstitution: 21.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 4559

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3684415, upper bound: 0.3636921
time: 7.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3668013, upper bound: 0.3653327
time: 4.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5415795, 0.5412340
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.7919369, 0.7857375
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6464705, 0.6364889
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6824224, 0.6877129
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7228465, 0.7151833
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6184444, 0.6254773
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6350951, 0.6429515
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6422315, 0.6371646
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5692043, 0.5746706
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6480236, 0.6505179

Time for backsubstitution: 21.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 4559

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3669317, upper bound: 0.3651985
time: 4.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6127

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3669342, upper bound: 0.3650474
time: 4.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3651405, upper bound: 0.3668422
time: 3.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5418825, 0.5409310
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.7866178, 0.7910461
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6497946, 0.6331592
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6797020, 0.6904302
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7213950, 0.7166324
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6158190, 0.6280956
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6313043, 0.6467366
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6455002, 0.6338849
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5667224, 0.5771456
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6432509, 0.6552787

Time for backsubstitution: 20.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4559
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 4559

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3674972, upper bound: 0.3646361
time: 4.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3658572, upper bound: 0.3662763
time: 4.73 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.19 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3662763, upper bound: 0.3658574
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3646361, upper bound: 0.3674977
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3668385, upper bound: 0.3652913
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3651982, upper bound: 0.3669322
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3653325, upper bound: 0.3668019
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3636920, upper bound: 0.3684416
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3658936, upper bound: 0.3662354
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3642536, upper bound: 0.3678677
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3678676, upper bound: 0.3642536
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3662351, upper bound: 0.3658937
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3684415, upper bound: 0.3636921
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3668013, upper bound: 0.3653327
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3669342, upper bound: 0.3650474
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3651405, upper bound: 0.3668422
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3674972, upper bound: 0.3646361
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.19
Output dim: 0, lower bound: -0.3658572, upper bound: 0.3662763

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

Time for backsubstitution: 20.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3662745, upper bound: 0.3640612
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3644811, upper bound: 0.3658560
time: 5.31 seconds

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

Time for backsubstitution: 20.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6127

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3646349, upper bound: 0.3657021
time: 8.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3628394, upper bound: 0.3674955
time: 4.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5367668, 0.5356314
1: -14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.7838430, 0.7889025
2: -7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6270256, 0.6393731
3: -8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.6911809, 0.6848929
4: -12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7106204, 0.7167666
5: -5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6203899, 0.6116662
6: -3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6305163, 0.6184883
7: -8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6247592, 0.6330292
8: -3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.5639122, 0.5548565
9: -2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6303449, 0.6203575

Time for backsubstitution: 20.97 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.39 + 546.61 = 605.00 seconds
