## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 7)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0177759909


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3979462, 0.3979462)
1: (-4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2524935, 0.2524936)
2: (-0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0159112, 0.0159112)
3: (-0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0627964, 0.0627964)
4: (-0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429302, 0.0429302)
5: (-0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0590519, 0.0590519)
6: (-1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0451884, 0.0451884)
7: (0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0387414, 0.0387414)
8: (-5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1682928, 0.1682928)
9: (-5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1521943, 0.1521943)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.45 + 269.48 = 276.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0178986, upper bound: 0.0178987

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3049

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178979, upper bound: 0.0179000
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178974, upper bound: 0.0179005
time: 3.30 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.69 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.69
Output dim: 5, lower bound: -0.0178979, upper bound: 0.0179000
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.69
Output dim: 5, lower bound: -0.0178974, upper bound: 0.0179005

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3979466, 0.3979470
1: -4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2524864, 0.2524871
2: -0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0159112, 0.0159112
3: -0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0627970, 0.0627968
4: -0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429342, 0.0429335
5: -0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0590502, 0.0590501
6: -1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0451884, 0.0451883
7: 0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0387236, 0.0387225
8: -5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1682892, 0.1682900
9: -5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1521821, 0.1521820

Time for backsubstitution: 5.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2599

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178965, upper bound: 0.0178918
time: 228.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178914, upper bound: 0.0178971
time: 104.49 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3979471, 0.3979466
1: -4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2524871, 0.2524864
2: -0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0159112, 0.0159112
3: -0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0627968, 0.0627970
4: -0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429335, 0.0429342
5: -0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0590501, 0.0590502
6: -1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0451883, 0.0451884
7: 0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0387225, 0.0387236
8: -5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1682899, 0.1682893
9: -5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1521820, 0.1521821

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2599

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178958, upper bound: 0.0178914
time: 29.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178909, upper bound: 0.0178968
time: 9.28 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 44.05 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 44.05
Output dim: 5, lower bound: -0.0178965, upper bound: 0.0178918
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 44.05
Output dim: 5, lower bound: -0.0178914, upper bound: 0.0178971
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 44.05
Output dim: 5, lower bound: -0.0178958, upper bound: 0.0178914
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 44.05
Output dim: 5, lower bound: -0.0178909, upper bound: 0.0178968

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3979385, 0.3979393
1: -4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2524757, 0.2524765
2: -0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0159051, 0.0159049
3: -0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0627969, 0.0627967
4: -0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429232, 0.0429224
5: -0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0590501, 0.0590500
6: -1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0451686, 0.0451682
7: 0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0386734, 0.0386746
8: -5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1682629, 0.1682643
9: -5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1521819, 0.1521817

Time for backsubstitution: 5.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2344

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177962, upper bound: 0.0178797
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178826, upper bound: 0.0177912
time: 49.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3979390, 0.3979388
1: -4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2524757, 0.2524765
2: -0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0159049, 0.0159051
3: -0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0627969, 0.0627967
4: -0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429231, 0.0429225
5: -0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0590501, 0.0590500
6: -1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0451683, 0.0451684
7: 0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0386757, 0.0386723
8: -5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1682635, 0.1682637
9: -5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1521819, 0.1521817

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2344

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177911, upper bound: 0.0178838
time: 22.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178775, upper bound: 0.0177970
time: 52.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3979390, 0.3979390
1: -4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2524765, 0.2524757
2: -0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0159051, 0.0159049
3: -0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0627967, 0.0627969
4: -0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429225, 0.0429231
5: -0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0590500, 0.0590501
6: -1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0451684, 0.0451683
7: 0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0386723, 0.0386757
8: -5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1682636, 0.1682635
9: -5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1521818, 0.1521819

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2344

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177955, upper bound: 0.0178799
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178818, upper bound: 0.0177931
time: 5.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3979393, 0.3979386
1: -4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2524765, 0.2524758
2: -0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0159049, 0.0159051
3: -0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0627967, 0.0627969
4: -0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429224, 0.0429232
5: -0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0590500, 0.0590501
6: -1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0451682, 0.0451686
7: 0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0386746, 0.0386734
8: -5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1682642, 0.1682629
9: -5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1521818, 0.1521819

Time for backsubstitution: 5.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2344

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177906, upper bound: 0.0178838
time: 117.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178770, upper bound: 0.0177978
time: 179.44 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 303.09 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 303.09
Output dim: 5, lower bound: -0.0177962, upper bound: 0.0178797
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 303.09
Output dim: 5, lower bound: -0.0178826, upper bound: 0.0177912
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 303.09
Output dim: 5, lower bound: -0.0177911, upper bound: 0.0178838
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 303.09
Output dim: 5, lower bound: -0.0178775, upper bound: 0.0177970
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 303.09
Output dim: 5, lower bound: -0.0177955, upper bound: 0.0178799
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 303.09
Output dim: 5, lower bound: -0.0178818, upper bound: 0.0177931
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 303.09
Output dim: 5, lower bound: -0.0177906, upper bound: 0.0178838
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 303.09
Output dim: 5, lower bound: -0.0178770, upper bound: 0.0177978

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3949090, 0.3948730
1: -4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2464921, 0.2463077
2: -0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0158373, 0.0158364
3: -0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0617955, 0.0618140
4: -0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429340, 0.0429333
5: -0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0578879, 0.0579072
6: -1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0445925, 0.0446199
7: 0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0386574, 0.0386586
8: -5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1670150, 0.1670043
9: -5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1505371, 0.1504829

Time for backsubstitution: 5.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2345

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177009, upper bound: 0.0177434
time: 38.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177377, upper bound: 0.0177456
time: 3.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3948722, 0.3949098
1: -4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2463069, 0.2464928
2: -0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0158366, 0.0158371
3: -0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0618142, 0.0617953
4: -0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429341, 0.0429332
5: -0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0579073, 0.0578878
6: -1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0446203, 0.0445921
7: 0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0386574, 0.0386586
8: -5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1670030, 0.1670163
9: -5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1504830, 0.1505370

Time for backsubstitution: 5.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2345

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177489, upper bound: 0.0177340
time: 5.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178419, upper bound: 0.0176952
time: 17.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3949094, 0.3948725
1: -4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2464920, 0.2463076
2: -0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0158371, 0.0158366
3: -0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0617955, 0.0618140
4: -0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429339, 0.0429334
5: -0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0578879, 0.0579072
6: -1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0445922, 0.0446201
7: 0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0386597, 0.0386564
8: -5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1670156, 0.1670037
9: -5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1505371, 0.1504829

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2345

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0176958, upper bound: 0.0176952
time: 76.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177326, upper bound: 0.0177500
time: 3.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3948725, 0.3949094
1: -4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2463069, 0.2464927
2: -0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0158364, 0.0158374
3: -0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0618142, 0.0617953
4: -0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429340, 0.0429333
5: -0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0579073, 0.0578879
6: -1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0446200, 0.0445924
7: 0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0386597, 0.0386563
8: -5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1670036, 0.1670157
9: -5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1504830, 0.1505370

Time for backsubstitution: 5.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2345

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177438, upper bound: 0.0177393
time: 39.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178368, upper bound: 0.0177012
time: 91.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3949094, 0.3948726
1: -4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2464927, 0.2463069
2: -0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0158374, 0.0158364
3: -0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0617954, 0.0618142
4: -0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429333, 0.0429340
5: -0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0578878, 0.0579073
6: -1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0445924, 0.0446200
7: 0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0386563, 0.0386597
8: -5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1670157, 0.1670036
9: -5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1505370, 0.1504830

Time for backsubstitution: 5.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2345

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177002, upper bound: 0.0176960
time: 29.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177369, upper bound: 0.0177461
time: 3.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3948724, 0.3949094
1: -4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2463076, 0.2464920
2: -0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0158366, 0.0158371
3: -0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0618140, 0.0617955
4: -0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429334, 0.0429339
5: -0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0579072, 0.0578879
6: -1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0446201, 0.0445922
7: 0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0386564, 0.0386597
8: -5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1670037, 0.1670156
9: -5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1504830, 0.1505371

Time for backsubstitution: 5.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2345

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177482, upper bound: 0.0177343
time: 57.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178412, upper bound: 0.0176954
time: 21.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3949099, 0.3948721
1: -4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2464928, 0.2463070
2: -0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0158372, 0.0158366
3: -0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0617953, 0.0618142
4: -0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429332, 0.0429341
5: -0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0578878, 0.0579073
6: -1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0445921, 0.0446203
7: 0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0386586, 0.0386574
8: -5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1670163, 0.1670030
9: -5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1505370, 0.1504830

Time for backsubstitution: 5.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2345

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0176953, upper bound: 0.0178441
time: 4.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177321, upper bound: 0.0177501
time: 4.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3948729, 0.3949091
1: -4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2463076, 0.2464920
2: -0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0158364, 0.0158373
3: -0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0618140, 0.0617955
4: -0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429333, 0.0429340
5: -0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0579072, 0.0578879
6: -1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0446199, 0.0445925
7: 0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0386586, 0.0386574
8: -5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1670043, 0.1670150
9: -5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1504830, 0.1505371

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2345

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177434, upper bound: 0.0177391
time: 4.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178363, upper bound: 0.0177028
time: 93.33 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 103.61 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0177009, upper bound: 0.0177434
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0177377, upper bound: 0.0177456
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0177489, upper bound: 0.0177340
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0178419, upper bound: 0.0176952
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0176958, upper bound: 0.0176952
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0177326, upper bound: 0.0177500
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0177438, upper bound: 0.0177393
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0178368, upper bound: 0.0177012
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0177002, upper bound: 0.0176960
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0177369, upper bound: 0.0177461
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0177482, upper bound: 0.0177343
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0178412, upper bound: 0.0176954
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0176953, upper bound: 0.0178441
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0177321, upper bound: 0.0177501
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0177434, upper bound: 0.0177391
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 103.61
Output dim: 5, lower bound: -0.0178363, upper bound: 0.0177028

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.3545628, -3.6957521, -4.3545628, -3.6957521, -0.3939569, 0.3940470
1: -4.7695675, -3.9107845, -4.7695675, -3.9107845, -0.2444054, 0.2446923
2: -0.4652619, -0.3604788, -0.4652619, -0.3604788, -0.0158168, 0.0158187
3: -0.3935674, -0.1224895, -0.3935674, -0.1224895, -0.0615184, 0.0614683
4: -0.2430127, 0.0103106, -0.2430127, 0.0103106, -0.0429171, 0.0429153
5: -0.0921660, 0.1835729, -0.0921660, 0.1835729, -0.0575564, 0.0575110
6: -1.3833758, -0.9588686, -1.3833758, -0.9588686, -0.0444596, 0.0444041
7: 0.3503961, 0.6162137, 0.3503961, 0.6162137, -0.0386510, 0.0386523
8: -5.1388440, -4.6581192, -5.1388440, -4.6581192, -0.1666040, 0.1666308
9: -5.0377841, -4.5070820, -5.0377841, -4.5070820, -0.1499486, 0.1500156

Time for backsubstitution: 5.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3518

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2567

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178279, upper bound: 0.0176735
time: 37.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178181, upper bound: 0.0176820
time: 98.58 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 142.23 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 142.23
Output dim: 5, lower bound: -0.0178279, upper bound: 0.0176735
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 142.23
Output dim: 5, lower bound: -0.0178181, upper bound: 0.0176820
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 142.23
Output dim: 5, lower bound: -0.0178368, upper bound: 0.0177012
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 142.23
Output dim: 5, lower bound: -0.0178412, upper bound: 0.0176954
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 142.23
Output dim: 5, lower bound: -0.0176953, upper bound: 0.0178441
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 142.23
Output dim: 5, lower bound: -0.0178363, upper bound: 0.0177028

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 276.93 + 1529.72 = 1806.65 seconds
