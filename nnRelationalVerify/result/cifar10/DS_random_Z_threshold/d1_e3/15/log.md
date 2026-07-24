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
execution time: IAR + RelationalAnalysis = 7.85 + 121.71 = 129.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0802878, upper bound: 0.0802884

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 801

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3330

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802873, upper bound: 0.0802807
time: 98.94 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802800, upper bound: 0.0802889
time: 141.48 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 240.43 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 240.43
Output dim: 7, lower bound: -0.0802873, upper bound: 0.0802807
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 240.43
Output dim: 7, lower bound: -0.0802800, upper bound: 0.0802889

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3884569, 0.3884721
1: -3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5711696, 0.5711396
2: -1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2013685, 0.2013946
3: -1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1249283, 0.1249359
4: 0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1755525, 0.1755624
5: -1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1327897, 0.1328109
6: -0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1605119, 0.1605062
7: 0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4625230, 0.4625215
8: -4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4119276, 0.4119489
9: -4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3454252, 0.3453882

Time for backsubstitution: 6.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2042

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3213

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802862, upper bound: 0.0802796
time: 156.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802866, upper bound: 0.0802788
time: 54.95 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3884721, 0.3884568
1: -3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5711396, 0.5711696
2: -1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2013946, 0.2013685
3: -1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1249359, 0.1249283
4: 0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1755624, 0.1755525
5: -1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1328109, 0.1327897
6: -0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1605062, 0.1605119
7: 0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4625215, 0.4625230
8: -4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4119489, 0.4119276
9: -4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3453882, 0.3454252

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 813

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2674

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802733, upper bound: 0.0802569
time: 206.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802494, upper bound: 0.0802831
time: 125.59 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 338.43 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 338.43
Output dim: 7, lower bound: -0.0802862, upper bound: 0.0802796
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 338.43
Output dim: 7, lower bound: -0.0802866, upper bound: 0.0802788
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 338.43
Output dim: 7, lower bound: -0.0802733, upper bound: 0.0802569
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 338.43
Output dim: 7, lower bound: -0.0802494, upper bound: 0.0802831

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3881461, 0.3881731
1: -3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5708300, 0.5708230
2: -1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2013574, 0.2013842
3: -1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1248293, 0.1248251
4: 0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1755357, 0.1755474
5: -1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1326911, 0.1326994
6: -0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1604618, 0.1604582
7: 0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4625196, 0.4625180
8: -4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4117032, 0.4117314
9: -4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3453097, 0.3452836

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 876

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802712, upper bound: 0.0802794
time: 22.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802860, upper bound: 0.0802653
time: 13.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3881578, 0.3881614
1: -3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5708530, 0.5707999
2: -1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2013580, 0.2013835
3: -1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1248175, 0.1248370
4: 0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1755376, 0.1755455
5: -1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1326782, 0.1327123
6: -0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1604640, 0.1604560
7: 0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4625194, 0.4625181
8: -4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4117101, 0.4117245
9: -4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3453206, 0.3452726

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3548

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 600

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802810, upper bound: 0.0802766
time: 267.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802833, upper bound: 0.0802740
time: 171.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3882994, 0.3883149
1: -3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5700701, 0.5703050
2: -1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2013617, 0.2013247
3: -1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1249216, 0.1249134
4: 0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1755442, 0.1755330
5: -1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1327841, 0.1327550
6: -0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1604468, 0.1604350
7: 0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4625048, 0.4625053
8: -4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4109804, 0.4111702
9: -4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3443187, 0.3445491

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2659

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 160

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802371, upper bound: 0.0802556
time: 17.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802727, upper bound: 0.0802127
time: 246.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3883302, 0.3882842
1: -3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5702749, 0.5701002
2: -1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2013509, 0.2013355
3: -1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1249210, 0.1249140
4: 0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1755428, 0.1755344
5: -1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1327762, 0.1327629
6: -0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1604293, 0.1604525
7: 0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4625038, 0.4625062
8: -4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4111916, 0.4109591
9: -4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3445120, 0.3443558

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 3332
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3500
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 3317
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 3571
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3548
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2864

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802491, upper bound: 0.0802828
time: 18.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802481, upper bound: 0.0802822
time: 123.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 148.51 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 148.51
Output dim: 7, lower bound: -0.0802712, upper bound: 0.0802794
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 148.51
Output dim: 7, lower bound: -0.0802860, upper bound: 0.0802653
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 148.51
Output dim: 7, lower bound: -0.0802810, upper bound: 0.0802766
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 148.51
Output dim: 7, lower bound: -0.0802833, upper bound: 0.0802740
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 148.51
Output dim: 7, lower bound: -0.0802371, upper bound: 0.0802556
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 148.51
Output dim: 7, lower bound: -0.0802727, upper bound: 0.0802127
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 148.51
Output dim: 7, lower bound: -0.0802491, upper bound: 0.0802828
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 148.51
Output dim: 7, lower bound: -0.0802481, upper bound: 0.0802822

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 129.56 + 1703.25 = 1832.81 seconds
