## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 15)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0802089108


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3889209, 0.3889208)
1: (-3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5720347, 0.5720347)
2: (-1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2019678, 0.2019678)
3: (-1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1251124, 0.1251123)
4: (0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1757070, 0.1757070)
5: (-1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1334287, 0.1334287)
6: (-0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1606191, 0.1606191)
7: (0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4625651, 0.4625652)
8: (-4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4125719, 0.4125719)
9: (-4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3464887, 0.3464887)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.81 + 122.11 = 129.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0802878, upper bound: 0.0802884

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3483

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802693, upper bound: 0.0802876
time: 27.29 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802874, upper bound: 0.0802712
time: 12.34 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 39.70 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 39.70
Output dim: 7, lower bound: -0.0802693, upper bound: 0.0802876
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 39.70
Output dim: 7, lower bound: -0.0802874, upper bound: 0.0802712

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3886213, 0.3886312
1: -3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5715552, 0.5715711
2: -1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2029918, 0.2029530
3: -1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1247158, 0.1247023
4: 0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1757289, 0.1757283
5: -1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1331696, 0.1331607
6: -0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1599297, 0.1599061
7: 0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4627299, 0.4627516
8: -4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4124065, 0.4124120
9: -4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3465199, 0.3465207

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3468

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802549, upper bound: 0.0802772
time: 11.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802573, upper bound: 0.0802719
time: 14.42 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3886313, 0.3886213
1: -3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5715710, 0.5715552
2: -1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2029529, 0.2029918
3: -1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1247023, 0.1247158
4: 0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1757283, 0.1757289
5: -1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1331607, 0.1331696
6: -0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1599061, 0.1599297
7: 0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4627516, 0.4627299
8: -4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4124120, 0.4124065
9: -4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3465207, 0.3465199

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3468

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802702, upper bound: 0.0802590
time: 34.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802754, upper bound: 0.0802549
time: 6.66 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 47.26 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 47.26
Output dim: 7, lower bound: -0.0802549, upper bound: 0.0802772
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 47.26
Output dim: 7, lower bound: -0.0802573, upper bound: 0.0802719
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 47.26
Output dim: 7, lower bound: -0.0802702, upper bound: 0.0802590
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 47.26
Output dim: 7, lower bound: -0.0802754, upper bound: 0.0802549

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3886195, 0.3886301
1: -3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5715510, 0.5715681
2: -1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2030044, 0.2029520
3: -1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1247130, 0.1246985
4: 0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1757293, 0.1757282
5: -1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1331681, 0.1331586
6: -0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1599264, 0.1598938
7: 0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4627202, 0.4627501
8: -4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4124056, 0.4124115
9: -4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3465191, 0.3465201

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3078

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0801802, upper bound: 0.0802749
time: 14.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802526, upper bound: 0.0802019
time: 326.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3886201, 0.3886294
1: -3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5715522, 0.5715669
2: -1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2029909, 0.2029655
3: -1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1247120, 0.1246994
4: 0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1757288, 0.1757288
5: -1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1331675, 0.1331592
6: -0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1599173, 0.1599028
7: 0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4627284, 0.4627419
8: -4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4124060, 0.4124112
9: -4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3465193, 0.3465199

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3078

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0801830, upper bound: 0.0802688
time: 14.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802549, upper bound: 0.0801964
time: 190.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3886294, 0.3886201
1: -3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5715668, 0.5715522
2: -1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2029655, 0.2029909
3: -1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1246994, 0.1247120
4: 0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1757288, 0.1757288
5: -1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1331592, 0.1331675
6: -0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1599028, 0.1599174
7: 0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4627419, 0.4627284
8: -4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4124112, 0.4124060
9: -4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3465199, 0.3465193

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3078

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0801957, upper bound: 0.0802560
time: 196.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802684, upper bound: 0.0801840
time: 75.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3886301, 0.3886195
1: -3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5715680, 0.5715510
2: -1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2029520, 0.2030044
3: -1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1246985, 0.1247130
4: 0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1757282, 0.1757293
5: -1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1331585, 0.1331681
6: -0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1598937, 0.1599264
7: 0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4627502, 0.4627202
8: -4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4124115, 0.4124056
9: -4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3465201, 0.3465191

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3078

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0802012, upper bound: 0.0801805
time: 161.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802729, upper bound: 0.0801795
time: 17.14 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 184.40 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 184.40
Output dim: 7, lower bound: -0.0801802, upper bound: 0.0802749
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 184.40
Output dim: 7, lower bound: -0.0802526, upper bound: 0.0802019
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 184.40
Output dim: 7, lower bound: -0.0801830, upper bound: 0.0802688
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 184.40
Output dim: 7, lower bound: -0.0802549, upper bound: 0.0801964
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 184.40
Output dim: 7, lower bound: -0.0801957, upper bound: 0.0802560
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 184.40
Output dim: 7, lower bound: -0.0802684, upper bound: 0.0801840
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 184.40
Output dim: 7, lower bound: -0.0802012, upper bound: 0.0801805
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 184.40
Output dim: 7, lower bound: -0.0802729, upper bound: 0.0801795

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3868610, 0.3868428
1: -3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5700040, 0.5699748
2: -1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2029455, 0.2028915
3: -1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1248310, 0.1248109
4: 0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1757001, 0.1756999
5: -1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1331841, 0.1331739
6: -0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1596824, 0.1596479
7: 0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4625176, 0.4625539
8: -4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4087153, 0.4086075
9: -4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3445971, 0.3445364

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2387

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0801300, upper bound: 0.0802713
time: 171.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0801767, upper bound: 0.0802230
time: 277.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3868322, 0.3868716
1: -3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5699577, 0.5700210
2: -1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2029439, 0.2028931
3: -1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1248255, 0.1248165
4: 0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1757010, 0.1756990
5: -1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1331835, 0.1331745
6: -0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1596806, 0.1596497
7: 0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4625240, 0.4625476
8: -4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4086016, 0.4087212
9: -4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3445355, 0.3445980

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2387

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0802011, upper bound: 0.0801989
time: 138.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802489, upper bound: 0.0801508
time: 73.11 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 217.66 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 217.66
Output dim: 7, lower bound: -0.0801300, upper bound: 0.0802713
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 217.66
Output dim: 7, lower bound: -0.0801767, upper bound: 0.0802230
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 217.66
Output dim: 7, lower bound: -0.0802011, upper bound: 0.0801989
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 217.66
Output dim: 7, lower bound: -0.0802489, upper bound: 0.0801508
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 217.66
Output dim: 7, lower bound: -0.0801830, upper bound: 0.0802688
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 217.66
Output dim: 7, lower bound: -0.0802549, upper bound: 0.0801964
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 217.66
Output dim: 7, lower bound: -0.0801957, upper bound: 0.0802560
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 217.66
Output dim: 7, lower bound: -0.0802684, upper bound: 0.0801840
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 217.66
Output dim: 7, lower bound: -0.0802729, upper bound: 0.0801795

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 129.92 + 1811.17 = 1941.09 seconds
