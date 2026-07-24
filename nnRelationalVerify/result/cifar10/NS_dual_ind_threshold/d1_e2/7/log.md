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
execution time: IAR + RelationalAnalysis = 7.27 + 278.75 = 286.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0178986, upper bound: 0.0178987

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 332
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2682
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2344
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 2836
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2309
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3036

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177779, upper bound: 0.0178151
time: 57.07 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178251, upper bound: 0.0178275
time: 7.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 64.22 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 64.22
Output dim: 5, lower bound: -0.0177779, upper bound: 0.0178151
NS_A2, status: Status.UNKNOWN, split count: 1, time: 64.22
Output dim: 5, lower bound: -0.0178251, upper bound: 0.0178275

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.3544550, -3.6985617, -4.3544769, -3.6980140, -0.3949229, 0.3943669
1: -4.7695675, -3.9142141, -4.7695675, -3.9135475, -0.2465068, 0.2454601
2: -0.4652603, -0.3605093, -0.4652606, -0.3605039, -0.0158690, 0.0158598
3: -0.3926677, -0.1224953, -0.3928456, -0.1224942, -0.0614791, 0.0616974
4: -0.2427654, 0.0103106, -0.2428064, 0.0103106, -0.0424152, 0.0424918
5: -0.0911416, 0.1835725, -0.0913432, 0.1835724, -0.0575635, 0.0577945
6: -1.3826344, -0.9588690, -1.3827817, -0.9588692, -0.0439256, 0.0441165
7: 0.3503995, 0.6162137, 0.3503988, 0.6162137, -0.0387318, 0.0387334
8: -5.1388440, -4.6586704, -5.1388440, -4.6585741, -0.1671033, 0.1668859
9: -5.0377841, -4.5077477, -5.0377841, -4.5076323, -0.1507202, 0.1504620

Time for backsubstitution: 5.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2682
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 2836
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2348

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177493, upper bound: 0.0176920
time: 49.39 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177493, upper bound: 0.0177859
time: 38.62 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.3564601, -3.6986432, -4.3544660, -3.6982069, -0.4029825, 0.3948170
1: -4.7712450, -3.9168537, -4.7695675, -3.9161143, -0.2632467, 0.2458393
2: -0.4652692, -0.3605241, -0.4652604, -0.3605165, -0.0159581, 0.0158725
3: -0.3926740, -0.1218248, -0.3928063, -0.1224948, -0.0616612, 0.0645641
4: -0.2426153, 0.0104691, -0.2426667, 0.0103106, -0.0424725, 0.0436075
5: -0.0911801, 0.1843468, -0.0913261, 0.1835726, -0.0577747, 0.0610585
6: -1.3824103, -0.9582009, -1.3825550, -0.9588690, -0.0439588, 0.0473898
7: 0.3504058, 0.6162117, 0.3504041, 0.6162137, -0.0387297, 0.0387451
8: -5.1387491, -4.6595521, -5.1388440, -4.6593456, -0.1703087, 0.1669421
9: -5.0377073, -4.5087404, -5.0377841, -4.5085011, -0.1546067, 0.1505370

Time for backsubstitution: 5.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2682
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 2836
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2348

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177964, upper bound: 0.0177054
time: 3.31 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177966, upper bound: 0.0177967
time: 199.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 208.59 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 208.59
Output dim: 5, lower bound: -0.0177493, upper bound: 0.0176920
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 208.59
Output dim: 5, lower bound: -0.0177493, upper bound: 0.0177859
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 208.59
Output dim: 5, lower bound: -0.0177964, upper bound: 0.0177054
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 208.59
Output dim: 5, lower bound: -0.0177966, upper bound: 0.0177967

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.3544245, -3.6997476, -4.3576593, -3.6993892, -0.3915868, 0.3987418
1: -4.7695675, -3.9167504, -4.7731209, -3.9164622, -0.2400577, 0.2541992
2: -0.4652598, -0.3605364, -0.4652541, -0.3605362, -0.0158451, 0.0158840
3: -0.3924187, -0.1224968, -0.3925622, -0.1213751, -0.0628855, 0.0606979
4: -0.2427038, 0.0103106, -0.2427356, 0.0108019, -0.0429657, 0.0421146
5: -0.0908420, 0.1835724, -0.0909960, 0.1848017, -0.0592356, 0.0566146
6: -1.3823204, -0.9588691, -1.3824258, -0.9575177, -0.0458537, 0.0428250
7: 0.3504193, 0.6162137, 0.3504230, 0.6161985, -0.0387390, 0.0387195
8: -5.1388435, -4.6598682, -5.1386094, -4.6599803, -0.1654322, 0.1689583
9: -5.0377841, -4.5086040, -5.0380578, -4.5086336, -0.1492359, 0.1523296

Time for backsubstitution: 5.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 332
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2682
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2344
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 2836
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2309
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 425

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177493, upper bound: 0.0176202
time: 286.24 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177494, upper bound: 0.0177866
time: 4.13 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.3563581, -3.7012362, -4.3543386, -3.7013137, -0.3990591, 0.3915292
1: -4.7712450, -3.9198420, -4.7695675, -3.9197667, -0.2565286, 0.2403536
2: -0.4652677, -0.3605314, -0.4652585, -0.3605255, -0.0159301, 0.0158491
3: -0.3919252, -0.1218303, -0.3918822, -0.1225016, -0.0607450, 0.0634554
4: -0.2422915, 0.0104691, -0.2422670, 0.0103106, -0.0421111, 0.0431779
5: -0.0903391, 0.1843463, -0.0903158, 0.1835718, -0.0567033, 0.0597682
6: -1.3814925, -0.9582014, -1.3814217, -0.9588694, -0.0428006, 0.0459829
7: 0.3504072, 0.6162116, 0.3504054, 0.6162137, -0.0387158, 0.0387295
8: -5.1387491, -4.6599321, -5.1388435, -4.6598148, -0.1686786, 0.1656083
9: -5.0377073, -4.5092878, -5.0377846, -4.5091524, -0.1531139, 0.1493003

Time for backsubstitution: 5.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 332
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2682
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2344
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 2836
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2309
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 425

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177966, upper bound: 0.0175397
time: 109.02 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177967, upper bound: 0.0177024
time: 32.25 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4.3564310, -3.6998284, -4.3576488, -3.6995819, -0.3996466, 0.3991920
1: -4.7712450, -3.9193900, -4.7731209, -3.9190292, -0.2567957, 0.2545793
2: -0.4652688, -0.3605512, -0.4652539, -0.3605487, -0.0159341, 0.0158967
3: -0.3924255, -0.1218265, -0.3925231, -0.1213756, -0.0630676, 0.0635643
4: -0.2425537, 0.0104691, -0.2425957, 0.0108019, -0.0430230, 0.0432300
5: -0.0908810, 0.1843466, -0.0909791, 0.1848017, -0.0594468, 0.0598779
6: -1.3820965, -0.9582008, -1.3821990, -0.9575182, -0.0458868, 0.0460982
7: 0.3504255, 0.6162117, 0.3504280, 0.6161985, -0.0387370, 0.0387312
8: -5.1387491, -4.6607494, -5.1386094, -4.6607509, -0.1686371, 0.1690146
9: -5.0377078, -4.5095968, -5.0380578, -4.5095019, -0.1531217, 0.1524045

Time for backsubstitution: 5.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 332
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2682
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2344
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 2836
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2309
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 425

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177965, upper bound: 0.0176349
time: 3.18 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177967, upper bound: 0.0177981
time: 101.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 110.22 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 110.22
Output dim: 5, lower bound: -0.0177493, upper bound: 0.0176202
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 110.22
Output dim: 5, lower bound: -0.0177494, upper bound: 0.0177866
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 110.22
Output dim: 5, lower bound: -0.0177966, upper bound: 0.0175397
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 110.22
Output dim: 5, lower bound: -0.0177967, upper bound: 0.0177024
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 110.22
Output dim: 5, lower bound: -0.0177965, upper bound: 0.0176349
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 110.22
Output dim: 5, lower bound: -0.0177967, upper bound: 0.0177981

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.3544211, -3.6982167, -4.3576555, -3.6993892, -0.3903413, 0.4002559
1: -4.7695584, -3.9159868, -4.7731118, -3.9164622, -0.2393958, 0.2550183
2: -0.4652473, -0.3603764, -0.4652426, -0.3605362, -0.0156886, 0.0160775
3: -0.3937892, -0.1225029, -0.3925622, -0.1213806, -0.0642970, 0.0595588
4: -0.2430302, 0.0102631, -0.2427354, 0.0107595, -0.0434051, 0.0417513
5: -0.0925461, 0.1835734, -0.0909960, 0.1848016, -0.0609378, 0.0552539
6: -1.3824679, -0.9590007, -1.3824258, -0.9576362, -0.0463875, 0.0423567
7: 0.3504203, 0.6163921, 0.3504236, 0.6161985, -0.0385930, 0.0389007
8: -5.1389370, -4.6598697, -5.1386094, -4.6599808, -0.1655241, 0.1688828
9: -5.0377755, -4.5086479, -5.0380578, -4.5086699, -0.1493083, 0.1522616

Time for backsubstitution: 5.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2682
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 2836
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3041

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177489, upper bound: 0.0176062
time: 59.71 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177486, upper bound: 0.0177849
time: 95.29 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.3549781, -3.7012362, -4.3532000, -3.7013137, -0.3976675, 0.3903829
1: -4.7705593, -3.9198420, -4.7690053, -3.9197667, -0.2557942, 0.2397505
2: -0.4651153, -0.3605314, -0.4651320, -0.3605255, -0.0157567, 0.0157073
3: -0.3919252, -0.1230593, -0.3918822, -0.1235378, -0.0597066, 0.0621842
4: -0.2422910, 0.0101429, -0.2422670, 0.0100412, -0.0417869, 0.0427835
5: -0.0903391, 0.1828210, -0.0903158, 0.1823274, -0.0554582, 0.0582428
6: -1.3814925, -0.9584248, -1.3814217, -0.9590744, -0.0424085, 0.0455025
7: 0.3505673, 0.6162113, 0.3505384, 0.6162134, -0.0385521, 0.0385956
8: -5.1387491, -4.6600165, -5.1388435, -4.6598849, -0.1686091, 0.1655238
9: -5.0377064, -4.5093145, -5.0377836, -4.5091743, -0.1530544, 0.1492293

Time for backsubstitution: 5.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2682
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 2836
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3041

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177961, upper bound: 0.0173591
time: 19.86 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177961, upper bound: 0.0175382
time: 50.34 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.3563538, -3.6997051, -4.3543348, -3.7013137, -0.3978136, 0.3930435
1: -4.7712359, -3.9190788, -4.7695589, -3.9197667, -0.2558667, 0.2411728
2: -0.4652551, -0.3603714, -0.4652470, -0.3605255, -0.0157735, 0.0160426
3: -0.3932958, -0.1218362, -0.3918822, -0.1225071, -0.0621565, 0.0623163
4: -0.2426180, 0.0104215, -0.2422670, 0.0102684, -0.0425506, 0.0428145
5: -0.0920432, 0.1843473, -0.0903158, 0.1835717, -0.0584055, 0.0584076
6: -1.3816400, -0.9583331, -1.3814217, -0.9589871, -0.0433344, 0.0455146
7: 0.3504081, 0.6163900, 0.3504063, 0.6162137, -0.0385698, 0.0389107
8: -5.1388431, -4.6599326, -5.1388435, -4.6598153, -0.1687706, 0.1655328
9: -5.0376997, -4.5093307, -5.0377846, -4.5091891, -0.1531862, 0.1492323

Time for backsubstitution: 5.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2682
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 2836
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3041

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177961, upper bound: 0.0175247
time: 5.97 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177962, upper bound: 0.0177022
time: 40.66 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.3550501, -3.6998284, -4.3565111, -3.6995819, -0.3982550, 0.3980460
1: -4.7705593, -3.9193900, -4.7725582, -3.9190292, -0.2560612, 0.2539760
2: -0.4651165, -0.3605512, -0.4651273, -0.3605487, -0.0157608, 0.0157549
3: -0.3924255, -0.1230553, -0.3925231, -0.1224120, -0.0620292, 0.0622931
4: -0.2425535, 0.0101429, -0.2425956, 0.0105323, -0.0426988, 0.0428356
5: -0.0908810, 0.1828212, -0.0909791, 0.1835570, -0.0582017, 0.0583524
6: -1.3820965, -0.9584244, -1.3821990, -0.9577234, -0.0454947, 0.0456178
7: 0.3505857, 0.6162114, 0.3505611, 0.6161984, -0.0385732, 0.0385973
8: -5.1387491, -4.6608343, -5.1386094, -4.6608219, -0.1685676, 0.1689301
9: -5.0377064, -4.5096240, -5.0380573, -4.5095243, -0.1530623, 0.1523335

Time for backsubstitution: 5.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2682
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 2836
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3041

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177960, upper bound: 0.0174541
time: 60.64 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177962, upper bound: 0.0176334
time: 3.89 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.3564267, -3.6982975, -4.3576460, -3.6995819, -0.3984010, 0.4007065
1: -4.7712359, -3.9186263, -4.7731118, -3.9190292, -0.2561337, 0.2553984
2: -0.4652563, -0.3603912, -0.4652424, -0.3605487, -0.0157776, 0.0160902
3: -0.3937961, -0.1218323, -0.3925231, -0.1213813, -0.0644791, 0.0624251
4: -0.2428800, 0.0104215, -0.2425958, 0.0107595, -0.0434625, 0.0428666
5: -0.0925850, 0.1843476, -0.0909791, 0.1848015, -0.0611490, 0.0585172
6: -1.3822440, -0.9583328, -1.3821990, -0.9576361, -0.0464206, 0.0456299
7: 0.3504265, 0.6163900, 0.3504288, 0.6161985, -0.0385910, 0.0389124
8: -5.1388431, -4.6607504, -5.1386094, -4.6607513, -0.1687290, 0.1689391
9: -5.0376997, -4.5096407, -5.0380578, -4.5095391, -0.1531941, 0.1523365

Time for backsubstitution: 5.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2334
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2682
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 2836
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3041

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177959, upper bound: 0.0174536
time: 82.06 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177960, upper bound: 0.0177980
time: 73.04 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 160.74 seconds
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 160.74
Output dim: 5, lower bound: -0.0177489, upper bound: 0.0176062
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 160.74
Output dim: 5, lower bound: -0.0177486, upper bound: 0.0177849
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 160.74
Output dim: 5, lower bound: -0.0177961, upper bound: 0.0173591
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 160.74
Output dim: 5, lower bound: -0.0177961, upper bound: 0.0175382
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 160.74
Output dim: 5, lower bound: -0.0177961, upper bound: 0.0175247
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 160.74
Output dim: 5, lower bound: -0.0177962, upper bound: 0.0177022
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 160.74
Output dim: 5, lower bound: -0.0177960, upper bound: 0.0174541
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 160.74
Output dim: 5, lower bound: -0.0177962, upper bound: 0.0176334
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 160.74
Output dim: 5, lower bound: -0.0177959, upper bound: 0.0174536
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 160.74
Output dim: 5, lower bound: -0.0177960, upper bound: 0.0177980

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.3544159, -3.6982343, -4.3576503, -3.6994107, -0.3813802, 0.4002016
1: -4.7695584, -3.9160049, -4.7731118, -3.9164841, -0.2278962, 0.2550082
2: -0.4652473, -0.3603888, -0.4652425, -0.3605513, -0.0155457, 0.0160745
3: -0.3937858, -0.1225033, -0.3925580, -0.1213809, -0.0642945, 0.0571727
4: -0.2430300, 0.0102507, -0.2427353, 0.0107443, -0.0433784, 0.0417491
5: -0.0925421, 0.1835731, -0.0909908, 0.1848015, -0.0609357, 0.0522561
6: -1.3824575, -0.9590008, -1.3824128, -0.9576361, -0.0463863, 0.0414551
7: 0.3504339, 0.6163921, 0.3504404, 0.6161986, -0.0385899, 0.0388140
8: -5.1389370, -4.6599464, -5.1386089, -4.6600604, -0.1562484, 0.1688285
9: -5.0377755, -4.5087523, -5.0380578, -4.5087757, -0.1434353, 0.1521827

Time for backsubstitution: 5.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2334
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 332
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2682
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2344
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 2836
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2309
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2589

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0176810, upper bound: 0.0177009
time: 87.33 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0176810, upper bound: 0.0177186
time: 3.68 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 286.02 + 1535.89 = 1821.92 seconds
