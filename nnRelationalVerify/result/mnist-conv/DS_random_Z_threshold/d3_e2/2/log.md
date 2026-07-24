## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.38370851399999995


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3888612, 1.3888612)
1: (-8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7070894, 0.7070894)
2: (10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6135144, 0.6135144)
3: (-7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2651591, 1.2651591)
4: (-7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1643662, 1.1643662)
5: (-13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.3121305, 1.3121300)
6: (-12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5428610, 1.5428610)
7: (-5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2241964, 1.2241964)
8: (-3.2843947, -2.1668067, -3.2843947, -2.1668067, -0.9989872, 0.9989872)
9: (-5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2323804, 1.2323804)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.22 + 35.16 = 58.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3915383, upper bound: 0.3915381

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4629
type: DSZ, layer: 1, pos: 6110

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4629

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3909413, upper bound: 0.3915375
time: 6.79 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915374, upper bound: 0.3909416
time: 6.38 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 13.18 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 13.18
Output dim: 2, lower bound: -0.3909413, upper bound: 0.3915375
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 13.18
Output dim: 2, lower bound: -0.3915374, upper bound: 0.3909416

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3731675, 1.3667731
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7041111, 0.7026174
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6111128, 0.6138725
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2652316, 1.2661228
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1198449, 1.1032782
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2902579, 1.2963243
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5466008, 1.5476952
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2130089, 1.2053046
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0024986, 1.0035305
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2122574, 1.2046366

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6110

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6110

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3905977, upper bound: 0.3915376
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3909404, upper bound: 0.3911999
time: 4.19 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3667731, 1.3731675
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7026174, 0.7041111
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6138725, 0.6111128
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2661228, 1.2652316
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1032782, 1.1198449
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2963243, 1.2902575
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5476947, 1.5466013
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2053046, 1.2130089
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0035305, 1.0024986
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2046366, 1.2122574

Time for backsubstitution: 22.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6110

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6110

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3909403
time: 12.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3905977
time: 8.45 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 43.69 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 43.69
Output dim: 2, lower bound: -0.3905977, upper bound: 0.3915376
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 43.69
Output dim: 2, lower bound: -0.3909404, upper bound: 0.3911999
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 43.69
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3909403
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 43.69
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3905977

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3705997, 1.3661695
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7026155, 0.7022660
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6107311, 0.6137807
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2632828, 1.2656655
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1190257, 1.0998254
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2900500, 1.2954521
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5449767, 1.5473108
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2090015, 1.2043643
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0020189, 1.0014915
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2119675, 1.2034073

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2380
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2607

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3890013, upper bound: 0.3802229
time: 6.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3793472, upper bound: 0.3900246
time: 5.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3725638, 1.3642054
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7037597, 0.7011218
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6110213, 0.6134908
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2647748, 1.2641740
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1163921, 1.1024590
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2893853, 1.2961173
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5462174, 1.5460711
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2120690, 1.2012968
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0004601, 1.0030508
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2110281, 1.2043467

Time for backsubstitution: 22.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2380
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 1096

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2607

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3893558, upper bound: 0.3798183
time: 4.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3797576, upper bound: 0.3896815
time: 4.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3642054, 1.3725638
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7011218, 0.7037599
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6134908, 0.6110213
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2641740, 1.2647748
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1024590, 1.1163921
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2961173, 1.2893853
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5460706, 1.5462179
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2012973, 1.2120690
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0030508, 1.0004601
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2043467, 1.2110281

Time for backsubstitution: 22.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2380
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1202

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 226

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3904766, upper bound: 0.3879143
time: 4.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3882263, upper bound: 0.3902088
time: 5.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3661695, 1.3705993
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7022660, 0.7026155
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6137807, 0.6107311
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2656655, 1.2632828
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.0998254, 1.1190257
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2954526, 1.2900505
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5473113, 1.5449772
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2043643, 1.2090015
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0014915, 1.0020189
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2034073, 1.2119675

Time for backsubstitution: 22.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 2380
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2607

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3900247, upper bound: 0.3793475
time: 7.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3802226, upper bound: 0.3890025
time: 4.15 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 34.14 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.14
Output dim: 2, lower bound: -0.3890013, upper bound: 0.3802229
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.14
Output dim: 2, lower bound: -0.3793472, upper bound: 0.3900246
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.14
Output dim: 2, lower bound: -0.3893558, upper bound: 0.3798183
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.14
Output dim: 2, lower bound: -0.3797576, upper bound: 0.3896815
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.14
Output dim: 2, lower bound: -0.3904766, upper bound: 0.3879143
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.14
Output dim: 2, lower bound: -0.3882263, upper bound: 0.3902088
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.14
Output dim: 2, lower bound: -0.3900247, upper bound: 0.3793475
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.14
Output dim: 2, lower bound: -0.3802226, upper bound: 0.3890025

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3679700, 1.3637171
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.6988263, 0.6986382
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6083415, 0.6085775
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2579646, 1.2635446
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1202297, 1.1010609
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2725978, 1.2830639
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5371318, 1.5365925
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.1629910, 1.1654367
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0007949, 1.0001898
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2084055, 1.1967282

Time for backsubstitution: 21.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2380
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 661

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1488

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3848612, upper bound: 0.3767557
time: 6.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3848576, upper bound: 0.3767664
time: 5.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3681273, 1.3635397
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.6989646, 0.6984768
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6055279, 0.6113758
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2611356, 1.2603474
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1202612, 1.1010251
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2776618, 1.2779684
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5342584, 1.5394382
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.1700058, 1.1583533
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0007172, 1.0002413
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2052884, 1.1998229

Time for backsubstitution: 21.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2380
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2822

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3773650, upper bound: 0.3872346
time: 4.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3766413, upper bound: 0.3876694
time: 4.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3699341, 1.3617330
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.6999705, 0.6974707
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6086164, 0.6082873
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2594562, 1.2620263
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1175914, 1.1036944
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2719016, 1.2837286
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5383449, 1.5353527
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.1660581, 1.1623011
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -0.9992094, 1.0017486
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2074437, 1.1976676

Time for backsubstitution: 22.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2380
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2462

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 306

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3891294, upper bound: 0.3781709
time: 5.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3875857, upper bound: 0.3795635
time: 6.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3701115, 1.3615756
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7001321, 0.6973324
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6058180, 0.6111009
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2626538, 1.2588553
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1176276, 1.1036630
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2769971, 1.2786646
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5354981, 1.5382252
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.1731415, 1.1552863
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -0.9991579, 1.0018268
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2043490, 1.2007847

Time for backsubstitution: 22.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2380
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1488

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1236

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3770542, upper bound: 0.3878301
time: 4.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3780233, upper bound: 0.3862785
time: 6.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3209524, 1.3288932
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.6625035, 0.6615272
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6042573, 0.6012305
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2560253, 1.2438641
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.0947280, 1.1215315
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2792435, 1.2838788
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5437407, 1.5441132
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.1894541, 1.1994157
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0035648, 1.0010591
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.1428542, 1.1699324

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 2380
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 1488

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3791103, upper bound: 0.3830291
time: 5.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3857373, upper bound: 0.3767623
time: 5.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3205347, 1.3293109
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.6588891, 0.6651416
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6036999, 0.6017880
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2432632, 1.2566261
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1075983, 1.1086612
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2906103, 1.2725110
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5439677, 1.5438871
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.1886439, 1.2002258
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0036497, 1.0009742
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.1632509, 1.1495352

Time for backsubstitution: 21.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 2380
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 306

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 661

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3851172, upper bound: 0.3886142
time: 4.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3869266, upper bound: 0.3873174
time: 7.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3635397, 1.3681273
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.6984768, 0.6989646
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6113758, 0.6055279
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2603474, 1.2611351
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1010251, 1.1202612
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2779689, 1.2776618
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5394387, 1.5342579
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.1583533, 1.1700058
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0002413, 1.0007167
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.1998229, 1.2052884

Time for backsubstitution: 21.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 2380
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 326

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1096

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3862969, upper bound: 0.3763762
time: 6.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3868487, upper bound: 0.3753439
time: 6.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3637171, 1.3679700
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.6986382, 0.6988263
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6085775, 0.6083415
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2635446, 1.2579646
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1010609, 1.1202297
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2830634, 1.2725978
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5365920, 1.5371313
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.1654367, 1.1629910
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0001898, 1.0007949
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.1967282, 1.2084055

Time for backsubstitution: 21.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2380
type: DSZ, layer: 3, pos: 1202

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2462

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3758154, upper bound: 0.3834949
time: 4.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3758187, upper bound: 0.3834677
time: 4.02 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.02 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3848612, upper bound: 0.3767557
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3848576, upper bound: 0.3767664
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3773650, upper bound: 0.3872346
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3766413, upper bound: 0.3876694
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3891294, upper bound: 0.3781709
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3875857, upper bound: 0.3795635
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3770542, upper bound: 0.3878301
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3780233, upper bound: 0.3862785
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3791103, upper bound: 0.3830291
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3857373, upper bound: 0.3767623
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3851172, upper bound: 0.3886142
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3869266, upper bound: 0.3873174
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3862969, upper bound: 0.3763762
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3868487, upper bound: 0.3753439
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3758154, upper bound: 0.3834949
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.02
Output dim: 2, lower bound: -0.3758187, upper bound: 0.3834677

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3724504, 1.3639441
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.6998734, 0.6967175
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6056869, 0.6106913
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2607951, 1.2680469
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1178403, 1.0958695
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2820230, 1.2864232
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5408034, 1.5452776
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2059650, 1.2010117
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0012021, 1.0005636
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2142191, 1.2009287

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2380

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1096

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3793296, upper bound: 0.3729063
time: 5.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3806665, upper bound: 0.3716613
time: 5.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3683739, 1.3661695
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7026155, 0.6995239
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6076417, 0.6137807
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2632828, 1.2631779
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1150699, 1.0998254
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2900500, 1.2874246
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5449767, 1.5431376
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2056489, 1.2043643
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0020189, 1.0006747
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2094889, 1.2034073

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2380
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 711

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2613

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3842257, upper bound: 0.3730064
time: 6.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3842043, upper bound: 0.3765268
time: 5.70 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 33.98 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 33.98
Output dim: 2, lower bound: -0.3793296, upper bound: 0.3729063
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 33.98
Output dim: 2, lower bound: -0.3806665, upper bound: 0.3716613
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 33.98
Output dim: 2, lower bound: -0.3842257, upper bound: 0.3730064
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 33.98
Output dim: 2, lower bound: -0.3842043, upper bound: 0.3765268
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 2, lower bound: -0.3773650, upper bound: 0.3872346
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 2, lower bound: -0.3766413, upper bound: 0.3876694
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 2, lower bound: -0.3891294, upper bound: 0.3781709
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 2, lower bound: -0.3875857, upper bound: 0.3795635
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 2, lower bound: -0.3770542, upper bound: 0.3878301
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 2, lower bound: -0.3780233, upper bound: 0.3862785
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 2, lower bound: -0.3857373, upper bound: 0.3767623
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 2, lower bound: -0.3851172, upper bound: 0.3886142
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 2, lower bound: -0.3869266, upper bound: 0.3873174
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 2, lower bound: -0.3862969, upper bound: 0.3763762
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 2, lower bound: -0.3868487, upper bound: 0.3753439

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.38 + 550.81 = 609.19 seconds
