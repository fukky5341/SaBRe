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
execution time: IAR + RelationalAnalysis = 23.38 + 36.86 = 60.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3915383, upper bound: 0.3915381

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4629
type: DSZ, layer: 1, pos: 6110

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 4629

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3909413, upper bound: 0.3915375
time: 7.62 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915374, upper bound: 0.3909416
time: 6.98 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 14.94 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 14.94
Output dim: 2, lower bound: -0.3909413, upper bound: 0.3915375
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 14.94
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

Time for backsubstitution: 20.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6110

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 6110

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3905977, upper bound: 0.3915376
time: 4.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3909404, upper bound: 0.3911999
time: 4.52 seconds

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

Time for backsubstitution: 22.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6110

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 6110

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3909403
time: 13.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3905977
time: 8.73 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 44.49 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 44.49
Output dim: 2, lower bound: -0.3905977, upper bound: 0.3915376
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 44.49
Output dim: 2, lower bound: -0.3909404, upper bound: 0.3911999
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 44.49
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3909403
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 44.49
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

Time for backsubstitution: 21.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2380

Time for candidate selection: 0.44 seconds

### Candidate
type: DSZ, layer: 3, pos: 2462

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3852867, upper bound: 0.3863846
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3853071, upper bound: 0.3863780
time: 6.04 seconds

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

Time for backsubstitution: 22.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2380

Time for candidate selection: 0.55 seconds

### Candidate
type: DSZ, layer: 3, pos: 2462

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3856481, upper bound: 0.3860222
time: 6.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3856690, upper bound: 0.3860168
time: 7.17 seconds

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

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2380

Time for candidate selection: 0.45 seconds

### Candidate
type: DSZ, layer: 3, pos: 2462

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3860158, upper bound: 0.3856691
time: 5.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3860220, upper bound: 0.3856483
time: 5.56 seconds

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

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2380

Time for candidate selection: 0.45 seconds

### Candidate
type: DSZ, layer: 3, pos: 2462

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3863776, upper bound: 0.3853082
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3863836, upper bound: 0.3852870
time: 5.21 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.45 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.45
Output dim: 2, lower bound: -0.3852867, upper bound: 0.3863846
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.45
Output dim: 2, lower bound: -0.3853071, upper bound: 0.3863780
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.45
Output dim: 2, lower bound: -0.3856481, upper bound: 0.3860222
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.45
Output dim: 2, lower bound: -0.3856690, upper bound: 0.3860168
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.45
Output dim: 2, lower bound: -0.3860158, upper bound: 0.3856691
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.45
Output dim: 2, lower bound: -0.3860220, upper bound: 0.3856483
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.45
Output dim: 2, lower bound: -0.3863776, upper bound: 0.3853082
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.45
Output dim: 2, lower bound: -0.3863836, upper bound: 0.3852870

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3691788, 1.3647742
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7024698, 0.7019341
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6106637, 0.6136997
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2612309, 1.2639813
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1198635, 1.0977464
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2891979, 1.2948785
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5458684, 1.5466528
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2063065, 1.1994333
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -0.9949131, 0.9981370
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2092195, 1.2075748

Time for backsubstitution: 22.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2380

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 326

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3823258, upper bound: 0.3833745
time: 9.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3811212, upper bound: 0.3840405
time: 6.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3705997, 1.3647485
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7022836, 0.7022660
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6106501, 0.6137807
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2615986, 1.2656655
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1169462, 1.0998254
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2894764, 1.2954521
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5443196, 1.5473108
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2090015, 1.2016687
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -0.9986639, 1.0014915
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2119675, 1.2006593

Time for backsubstitution: 21.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2380

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 326

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3824264, upper bound: 0.3833310
time: 5.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3811838, upper bound: 0.3839791
time: 4.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3711429, 1.3628101
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7036140, 0.7007899
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6109538, 0.6134095
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2627225, 1.2624893
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1172304, 1.1003799
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2885332, 1.2955437
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5471101, 1.5454130
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2093735, 1.1963663
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -0.9933543, 0.9996958
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2082801, 1.2085142

Time for backsubstitution: 22.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2380

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 326

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3826829, upper bound: 0.3830174
time: 7.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3814768, upper bound: 0.3836825
time: 7.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3725638, 1.3627844
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7034280, 0.7011218
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6109400, 0.6134908
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2630901, 1.2641740
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1143131, 1.1024590
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2888117, 1.2961173
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5455594, 1.5460711
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2120690, 1.1986017
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -0.9971051, 1.0030508
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2110281, 1.2015986

Time for backsubstitution: 22.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2380

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 326

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3827840, upper bound: 0.3829751
time: 6.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3815413, upper bound: 0.3836213
time: 4.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3627844, 1.3711686
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7009759, 0.7034280
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6134231, 0.6109402
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2621217, 1.2630901
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1032972, 1.1143131
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2952642, 1.2888117
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5469632, 1.5455599
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.1986017, 1.2071381
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -0.9959450, 0.9971051
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2015986, 1.2151957

Time for backsubstitution: 22.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2380

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 3, pos: 326

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3836202, upper bound: 0.3815424
time: 4.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3829750, upper bound: 0.3827843
time: 7.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3642054, 1.3711429
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7007899, 0.7037599
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6134095, 0.6110213
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2624893, 1.2647748
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1003799, 1.1163921
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2955437, 1.2893853
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5454125, 1.5462179
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2012973, 1.2093735
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -0.9996958, 1.0004601
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2043467, 1.2082801

Time for backsubstitution: 22.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2380

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 3, pos: 326

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3836822, upper bound: 0.3814779
time: 4.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3830173, upper bound: 0.3826840
time: 5.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3647485, 1.3692045
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7021203, 0.7022836
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6137133, 0.6106501
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2636137, 1.2615986
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1006637, 1.1169467
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2945995, 1.2894769
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5482030, 1.5443192
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2016687, 1.2040706
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -0.9943862, 0.9986639
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2006593, 1.2161350

Time for backsubstitution: 22.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2380

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 326

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3839780, upper bound: 0.3811840
time: 6.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3833308, upper bound: 0.3824267
time: 6.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3661695, 1.3691788
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7019341, 0.7026155
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6136997, 0.6107311
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2639813, 1.2632828
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.0977464, 1.1190257
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2948790, 1.2900505
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5466523, 1.5449772
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2043643, 1.2063065
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -0.9981370, 1.0020189
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2034073, 1.2092195

Time for backsubstitution: 22.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2380

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 326

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3840402, upper bound: 0.3811223
time: 4.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3833746, upper bound: 0.3823262
time: 7.49 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 34.18 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3823258, upper bound: 0.3833745
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3811212, upper bound: 0.3840405
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3824264, upper bound: 0.3833310
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3811838, upper bound: 0.3839791
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3826829, upper bound: 0.3830174
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3814768, upper bound: 0.3836825
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3827840, upper bound: 0.3829751
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3815413, upper bound: 0.3836213
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3836202, upper bound: 0.3815424
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3829750, upper bound: 0.3827843
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3836822, upper bound: 0.3814779
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3830173, upper bound: 0.3826840
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3839780, upper bound: 0.3811840
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3833308, upper bound: 0.3824267
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3840402, upper bound: 0.3811223
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.18
Output dim: 2, lower bound: -0.3833746, upper bound: 0.3823262

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3637853, 1.3646569
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7019961, 0.7011285
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6091824, 0.6129146
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2628493, 1.2647610
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1174593, 1.0982199
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2857480, 1.2920332
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5397897, 1.5412169
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2065921, 1.2004733
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0003066, 0.9997587
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2104783, 1.2012482

Time for backsubstitution: 22.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2462
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 663
type: DSZ, layer: 3, pos: 1090
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2140
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 306
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2511
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2633
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1229
type: DSZ, layer: 3, pos: 2847
type: DSZ, layer: 3, pos: 2822
type: DSZ, layer: 3, pos: 3122
type: DSZ, layer: 3, pos: 2380

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2462

### Candidate
type: DSZ, layer: 3, pos: 1096

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3771530, upper bound: 0.3804199
time: 6.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3775002, upper bound: 0.3800502
time: 6.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3637853, 1.3646569
1: -8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7019961, 0.7011285
2: 10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6091824, 0.6129146
3: -7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2628493, 1.2647610
4: -7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1174593, 1.0982199
5: -13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.2857480, 1.2920332
6: -12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5397897, 1.5412169
7: -5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2065921, 1.2004733
8: -3.2843947, -2.1668067, -3.2843947, -2.1668067, -1.0003066, 0.9997587
9: -5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2104783, 1.2012482

Time for backsubstitution: 22.17 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 60.24 + 559.60 = 619.84 seconds
