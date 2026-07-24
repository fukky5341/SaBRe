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
execution time: IAR + RelationalAnalysis = 7.18 + 273.17 = 280.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0178986, upper bound: 0.0178987

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 3036
type: A, layer: 1, pos: 425
type: B, layer: 1, pos: 425
type: A, layer: 1, pos: 3041
type: B, layer: 1, pos: 3041
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: B, layer: 1, pos: 2345
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 2169
type: B, layer: 1, pos: 2169
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2553
type: B, layer: 1, pos: 2553
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 2523
type: B, layer: 1, pos: 2523
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 2767
type: B, layer: 1, pos: 2767
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 3468
type: B, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2794
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 3092
type: B, layer: 1, pos: 3092
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 366
type: B, layer: 1, pos: 366
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2367
type: B, layer: 1, pos: 2367
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 2344
type: A, layer: 1, pos: 2797
type: B, layer: 1, pos: 2797
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3111
type: B, layer: 1, pos: 3111
type: A, layer: 1, pos: 3093
type: B, layer: 1, pos: 3093
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 3049
type: B, layer: 1, pos: 3049
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 3132
type: B, layer: 1, pos: 3132
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 2643
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 3244
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 2610
type: B, layer: 1, pos: 2610
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 2214
type: B, layer: 1, pos: 2214
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2348

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177760, upper bound: 0.0178715
time: 3.68 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0178699, upper bound: 0.0178725
time: 4.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.35 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.35
Output dim: 5, lower bound: -0.0177760, upper bound: 0.0178715
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.35
Output dim: 5, lower bound: -0.0178699, upper bound: 0.0178725

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.3544350, -3.6988587, -4.3544598, -3.6983449, -0.3946579, 0.3940223
1: -4.7695675, -3.9144368, -4.7695675, -3.9137731, -0.2470074, 0.2457738
2: -0.4652600, -0.3604878, -0.4652603, -0.3604861, -0.0158878, 0.0158832
3: -0.3926432, -0.1224963, -0.3928185, -0.1224949, -0.0616873, 0.0618803
4: -0.2426131, 0.0103106, -0.2426885, 0.0103106, -0.0425003, 0.0425689
5: -0.0911558, 0.1835723, -0.0913250, 0.1835726, -0.0577610, 0.0579804
6: -1.3822428, -0.9588690, -1.3824576, -0.9588689, -0.0437815, 0.0440303
7: 0.3503975, 0.6162137, 0.3503973, 0.6162137, -0.0387258, 0.0387274
8: -5.1388435, -4.6585884, -5.1388435, -4.6584997, -0.1669589, 0.1666622
9: -5.0377846, -4.5077338, -5.0377841, -4.5076299, -0.1509576, 0.1507010

Time for backsubstitution: 5.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3036
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 425
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 3041
type: B, layer: 1, pos: 3041
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2345
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 2169
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 2137
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2553
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 2523
type: A, layer: 1, pos: 2523
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 2767
type: B, layer: 1, pos: 2767
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 3468
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 2794
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 3092
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 366
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 2367
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 2344
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3111
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 3093
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 3049
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3132
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 2643
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3244
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2610
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2214
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3036

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0176905, upper bound: 0.0177504
time: 72.20 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177025, upper bound: 0.0177982
time: 61.22 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.3577452, -3.6971259, -4.3545322, -3.6969366, -0.4023213, 0.3946103
1: -4.7731209, -3.9136987, -4.7695675, -3.9133201, -0.2612335, 0.2460409
2: -0.4652553, -0.3605110, -0.4652614, -0.3605059, -0.0159354, 0.0158873
3: -0.3932845, -0.1213703, -0.3933191, -0.1224912, -0.0617965, 0.0642030
4: -0.2429417, 0.0108019, -0.2429507, 0.0103106, -0.0425523, 0.0434808
5: -0.0918191, 0.1848021, -0.0918671, 0.1835726, -0.0578706, 0.0607241
6: -1.3830199, -0.9575179, -1.3830616, -0.9588689, -0.0438968, 0.0471165
7: 0.3504201, 0.6161985, 0.3504158, 0.6162137, -0.0387275, 0.0387486
8: -5.1386094, -4.6595244, -5.1388435, -4.6593170, -0.1703652, 0.1666208
9: -5.0380578, -4.5080833, -5.0377841, -4.5079389, -0.1540620, 0.1507089

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3036
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 425
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 3041
type: B, layer: 1, pos: 3041
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2345
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2169
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 2137
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2553
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 2523
type: A, layer: 1, pos: 2523
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 2767
type: B, layer: 1, pos: 2767
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 3468
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 2794
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 3092
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 366
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 2367
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 2797
type: A, layer: 1, pos: 2797
type: B, layer: 1, pos: 2344
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3111
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 3093
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 3049
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 3132
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 2643
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 3244
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2610
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2214
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3036

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177848, upper bound: 0.0177489
time: 65.56 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177967, upper bound: 0.0177995
time: 3.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 74.17 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 74.17
Output dim: 5, lower bound: -0.0176905, upper bound: 0.0177504
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 74.17
Output dim: 5, lower bound: -0.0177025, upper bound: 0.0177982
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 74.17
Output dim: 5, lower bound: -0.0177848, upper bound: 0.0177489
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 74.17
Output dim: 5, lower bound: -0.0177967, upper bound: 0.0177995

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.3543386, -3.7013137, -4.3563581, -3.7012362, -0.3915293, 0.3990589
1: -4.7695675, -3.9197667, -4.7712450, -3.9198420, -0.2403536, 0.2565287
2: -0.4652585, -0.3605255, -0.4652677, -0.3605314, -0.0158491, 0.0159301
3: -0.3918822, -0.1225016, -0.3919252, -0.1218303, -0.0634554, 0.0607451
4: -0.2422670, 0.0103106, -0.2422915, 0.0104691, -0.0431779, 0.0421111
5: -0.0903158, 0.1835718, -0.0903391, 0.1843463, -0.0597682, 0.0567033
6: -1.3814217, -0.9588694, -1.3814925, -0.9582014, -0.0459829, 0.0428006
7: 0.3504054, 0.6162137, 0.3504072, 0.6162116, -0.0387295, 0.0387158
8: -5.1388435, -4.6598148, -5.1387491, -4.6599321, -0.1656082, 0.1686786
9: -5.0377846, -4.5091524, -5.0377073, -4.5092878, -0.1493003, 0.1531139

Time for backsubstitution: 5.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 425
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 3041
type: A, layer: 1, pos: 3041
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 2169
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2553
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 2523
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 2767
type: A, layer: 1, pos: 2767
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3468
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2794
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 3092
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 366
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2367
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2344
type: A, layer: 1, pos: 2797
type: B, layer: 1, pos: 2797
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3111
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 3093
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 3049
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3132
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 3244
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2610
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2214
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 425

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177024, upper bound: 0.0176338
time: 4.64 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177025, upper bound: 0.0177986
time: 3.67 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.3576593, -3.6993892, -4.3544245, -3.6997476, -0.3987418, 0.3915867
1: -4.7731209, -3.9164622, -4.7695675, -3.9167504, -0.2541992, 0.2400577
2: -0.4652541, -0.3605362, -0.4652598, -0.3605364, -0.0158840, 0.0158451
3: -0.3925622, -0.1213751, -0.3924187, -0.1224968, -0.0606979, 0.0628854
4: -0.2427356, 0.0108019, -0.2427038, 0.0103106, -0.0421146, 0.0429657
5: -0.0909960, 0.1848017, -0.0908420, 0.1835724, -0.0566146, 0.0592356
6: -1.3824258, -0.9575177, -1.3823204, -0.9588691, -0.0428250, 0.0458537
7: 0.3504230, 0.6161985, 0.3504193, 0.6162137, -0.0387195, 0.0387390
8: -5.1386094, -4.6599803, -5.1388435, -4.6598682, -0.1689583, 0.1654322
9: -5.0380578, -4.5086336, -5.0377841, -4.5086040, -0.1523296, 0.1492359

Time for backsubstitution: 5.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 425
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 3041
type: B, layer: 1, pos: 3041
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: B, layer: 1, pos: 2345
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2169
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2553
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 2523
type: A, layer: 1, pos: 2523
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 2767
type: B, layer: 1, pos: 2767
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 3468
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 2794
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 3092
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 366
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 2367
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 2797
type: A, layer: 1, pos: 2797
type: B, layer: 1, pos: 2344
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3111
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 3093
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 3049
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 3132
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 2643
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 3244
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2610
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2214
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 425

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0176203, upper bound: 0.0177501
time: 149.66 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177846, upper bound: 0.0177513
time: 3.35 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4.3576488, -3.6995819, -4.3564310, -3.6998284, -0.3991920, 0.3996466
1: -4.7731209, -3.9190292, -4.7712450, -3.9193900, -0.2545792, 0.2567957
2: -0.4652539, -0.3605487, -0.4652688, -0.3605512, -0.0158967, 0.0159341
3: -0.3925231, -0.1213756, -0.3924255, -0.1218265, -0.0635642, 0.0630676
4: -0.2425957, 0.0108019, -0.2425537, 0.0104691, -0.0432300, 0.0430230
5: -0.0909791, 0.1848017, -0.0908810, 0.1843466, -0.0598779, 0.0594468
6: -1.3821990, -0.9575182, -1.3820965, -0.9582008, -0.0460982, 0.0458868
7: 0.3504280, 0.6161985, 0.3504255, 0.6162117, -0.0387312, 0.0387370
8: -5.1386094, -4.6607509, -5.1387491, -4.6607494, -0.1690146, 0.1686371
9: -5.0380578, -4.5095019, -5.0377078, -4.5095968, -0.1524046, 0.1531217

Time for backsubstitution: 5.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 425
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 3041
type: A, layer: 1, pos: 3041
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 2169
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2553
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 2523
type: B, layer: 1, pos: 2523
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 2767
type: A, layer: 1, pos: 2767
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3468
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2794
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 3092
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 366
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2367
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2344
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3111
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 3093
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 3049
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3132
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 3244
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2610
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2214
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 425

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177965, upper bound: 0.0176331
time: 73.81 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177966, upper bound: 0.0177988
time: 3.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 82.82 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 82.82
Output dim: 5, lower bound: -0.0177024, upper bound: 0.0176338
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 82.82
Output dim: 5, lower bound: -0.0177025, upper bound: 0.0177986
NS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 82.82
Output dim: 5, lower bound: -0.0176203, upper bound: 0.0177501
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 82.82
Output dim: 5, lower bound: -0.0177846, upper bound: 0.0177513
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 82.82
Output dim: 5, lower bound: -0.0177965, upper bound: 0.0176331
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 82.82
Output dim: 5, lower bound: -0.0177966, upper bound: 0.0177988

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.3543344, -3.6997831, -4.3563547, -3.7012362, -0.3902837, 0.4005733
1: -4.7695584, -3.9190030, -4.7712374, -3.9198420, -0.2396918, 0.2573477
2: -0.4652460, -0.3603654, -0.4652562, -0.3605314, -0.0156925, 0.0161236
3: -0.3932528, -0.1225075, -0.3919252, -0.1218359, -0.0648670, 0.0596059
4: -0.2425933, 0.0102631, -0.2422913, 0.0104267, -0.0436173, 0.0417478
5: -0.0920199, 0.1835729, -0.0903391, 0.1843463, -0.0614704, 0.0553426
6: -1.3815696, -0.9590011, -1.3814925, -0.9583195, -0.0465167, 0.0423323
7: 0.3504065, 0.6163921, 0.3504079, 0.6162116, -0.0385835, 0.0388970
8: -5.1389370, -4.6598148, -5.1387491, -4.6599331, -0.1657002, 0.1686031
9: -5.0377760, -4.5091963, -5.0377073, -4.5093241, -0.1493727, 0.1530458

Time for backsubstitution: 5.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3041
type: A, layer: 1, pos: 3041
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2169
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2553
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 2523
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 2767
type: A, layer: 1, pos: 2767
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3468
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2794
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 3092
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 366
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2367
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2344
type: A, layer: 1, pos: 2797
type: B, layer: 1, pos: 2797
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3111
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 3093
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 3049
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 3132
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 3244
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2610
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2214
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3041

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177020, upper bound: 0.0176175
time: 13.21 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177018, upper bound: 0.0177968
time: 6.55 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -4.3576555, -3.6993892, -4.3544211, -3.6982167, -0.4002559, 0.3903414
1: -4.7731118, -3.9164622, -4.7695584, -3.9159868, -0.2550184, 0.2393958
2: -0.4652426, -0.3605362, -0.4652473, -0.3603764, -0.0160775, 0.0156886
3: -0.3925622, -0.1213806, -0.3937892, -0.1225029, -0.0595588, 0.0642970
4: -0.2427354, 0.0107595, -0.2430302, 0.0102631, -0.0417513, 0.0434051
5: -0.0909960, 0.1848016, -0.0925461, 0.1835734, -0.0552539, 0.0609378
6: -1.3824258, -0.9576362, -1.3824679, -0.9590007, -0.0423567, 0.0463875
7: 0.3504236, 0.6161985, 0.3504203, 0.6163921, -0.0389007, 0.0385930
8: -5.1386094, -4.6599808, -5.1389370, -4.6598697, -0.1688828, 0.1655241
9: -5.0380578, -4.5086699, -5.0377755, -4.5086479, -0.1522616, 0.1493083

Time for backsubstitution: 5.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3041
type: B, layer: 1, pos: 3041
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: B, layer: 1, pos: 2345
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 2169
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2553
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 2523
type: A, layer: 1, pos: 2523
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 2767
type: B, layer: 1, pos: 2767
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 3468
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 2794
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 3092
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 366
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 2367
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 2797
type: A, layer: 1, pos: 2797
type: B, layer: 1, pos: 2344
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3111
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 3093
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 3049
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 3132
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 2643
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 3244
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2610
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2214
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3041

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0176046, upper bound: 0.0177498
time: 62.45 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177842, upper bound: 0.0177509
time: 3.27 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.3562694, -3.6995819, -4.3552923, -3.6998284, -0.3978007, 0.3985002
1: -4.7724352, -3.9190292, -4.7706833, -3.9193900, -0.2538448, 0.2561924
2: -0.4651016, -0.3605487, -0.4651422, -0.3605512, -0.0157233, 0.0157924
3: -0.3925231, -0.1226047, -0.3924255, -0.1228626, -0.0625258, 0.0617964
4: -0.2425957, 0.0104758, -0.2425534, 0.0101994, -0.0429057, 0.0426286
5: -0.0909791, 0.1832762, -0.0908810, 0.1831021, -0.0586327, 0.0579214
6: -1.3821990, -0.9577415, -1.3820965, -0.9584060, -0.0457061, 0.0454064
7: 0.3505882, 0.6161983, 0.3505586, 0.6162114, -0.0385675, 0.0386030
8: -5.1386094, -4.6608353, -5.1387491, -4.6608200, -0.1689450, 0.1685527
9: -5.0380569, -4.5095296, -5.0377073, -4.5096188, -0.1523451, 0.1530508

Time for backsubstitution: 5.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3041
type: A, layer: 1, pos: 3041
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 2169
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2553
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 2523
type: B, layer: 1, pos: 2523
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 2767
type: A, layer: 1, pos: 2767
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3468
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2794
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 3092
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 366
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2367
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2344
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3111
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 3093
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 3049
type: A, layer: 1, pos: 3049
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3132
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 3244
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2610
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2214
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3041

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177961, upper bound: 0.0174530
time: 126.66 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177958, upper bound: 0.0176337
time: 3.90 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.3576450, -3.6980507, -4.3564272, -3.6998284, -0.3979464, 0.4011608
1: -4.7731113, -3.9182656, -4.7712374, -3.9193900, -0.2539173, 0.2576148
2: -0.4652414, -0.3603887, -0.4652573, -0.3605512, -0.0157401, 0.0161277
3: -0.3938937, -0.1213815, -0.3924255, -0.1218320, -0.0649758, 0.0619285
4: -0.2429221, 0.0107542, -0.2425536, 0.0104267, -0.0436694, 0.0426597
5: -0.0926831, 0.1848027, -0.0908810, 0.1843466, -0.0615801, 0.0580861
6: -1.3823467, -0.9576501, -1.3820965, -0.9583189, -0.0466320, 0.0454185
7: 0.3504290, 0.6163771, 0.3504264, 0.6162117, -0.0385852, 0.0389181
8: -5.1387033, -4.6607513, -5.1387491, -4.6607504, -0.1691064, 0.1685617
9: -5.0380497, -4.5095463, -5.0377078, -4.5096335, -0.1524769, 0.1530537

Time for backsubstitution: 5.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3041
type: A, layer: 1, pos: 3041
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 2169
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2553
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 2523
type: B, layer: 1, pos: 2523
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 2767
type: A, layer: 1, pos: 2767
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3468
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2794
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 3092
type: B, layer: 1, pos: 3092
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 366
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2367
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2344
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3111
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 3093
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 3049
type: A, layer: 1, pos: 3049
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3132
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 3244
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2610
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2214
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3041

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177958, upper bound: 0.0176184
time: 4.54 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0177961, upper bound: 0.0177971
time: 46.87 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 57.04 seconds
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 57.04
Output dim: 5, lower bound: -0.0177020, upper bound: 0.0176175
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 57.04
Output dim: 5, lower bound: -0.0177018, upper bound: 0.0177968
NS_A2_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 57.04
Output dim: 5, lower bound: -0.0176046, upper bound: 0.0177498
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 57.04
Output dim: 5, lower bound: -0.0177842, upper bound: 0.0177509
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 57.04
Output dim: 5, lower bound: -0.0177961, upper bound: 0.0174530
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 57.04
Output dim: 5, lower bound: -0.0177958, upper bound: 0.0176337
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 57.04
Output dim: 5, lower bound: -0.0177958, upper bound: 0.0176184
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 57.04
Output dim: 5, lower bound: -0.0177961, upper bound: 0.0177971

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.3543296, -3.6998005, -4.3563485, -3.7012579, -0.3813224, 0.4005187
1: -4.7695584, -3.9190211, -4.7712369, -3.9198647, -0.2281922, 0.2573377
2: -0.4652459, -0.3603779, -0.4652561, -0.3605466, -0.0155496, 0.0161206
3: -0.3932494, -0.1225078, -0.3919210, -0.1218362, -0.0648644, 0.0572198
4: -0.2425932, 0.0102507, -0.2422913, 0.0104117, -0.0435906, 0.0417456
5: -0.0920157, 0.1835727, -0.0903339, 0.1843461, -0.0614683, 0.0523447
6: -1.3815593, -0.9590009, -1.3814796, -0.9583192, -0.0465155, 0.0414307
7: 0.3504201, 0.6163921, 0.3504246, 0.6162117, -0.0385805, 0.0388103
8: -5.1389370, -4.6598921, -5.1387491, -4.6600127, -0.1564245, 0.1685488
9: -5.0377755, -4.5093002, -5.0377073, -4.5094304, -0.1434996, 0.1529670

Time for backsubstitution: 5.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 2169
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2553
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 2523
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 2767
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 3041
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3468
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2794
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 3092
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 366
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2367
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2344
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2797
type: B, layer: 1, pos: 2797
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3111
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 3093
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 3049
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 3132
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 3244
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2610
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2214
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 2589

## Relational analysis of NS_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0176114, upper bound: 0.0177312
time: 3.08 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0176393, upper bound: 0.0177285
time: 26.40 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -4.3576503, -3.6994107, -4.3544159, -3.6982343, -0.4002016, 0.3813801
1: -4.7731118, -3.9164841, -4.7695584, -3.9160049, -0.2550082, 0.2278962
2: -0.4652425, -0.3605513, -0.4652473, -0.3603888, -0.0160745, 0.0155457
3: -0.3925580, -0.1213809, -0.3937858, -0.1225033, -0.0571727, 0.0642944
4: -0.2427353, 0.0107443, -0.2430300, 0.0102507, -0.0417491, 0.0433784
5: -0.0909908, 0.1848015, -0.0925421, 0.1835731, -0.0522561, 0.0609357
6: -1.3824128, -0.9576361, -1.3824575, -0.9590008, -0.0414551, 0.0463863
7: 0.3504404, 0.6161986, 0.3504339, 0.6163921, -0.0388140, 0.0385899
8: -5.1386089, -4.6600604, -5.1389370, -4.6599464, -0.1688284, 0.1562483
9: -5.0380578, -4.5087757, -5.0377755, -4.5087523, -0.1521826, 0.1434353

Time for backsubstitution: 5.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2345
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 2169
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2553
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 2523
type: A, layer: 1, pos: 2523
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 2767
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 3041
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 3468
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 2794
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 3092
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 366
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 2367
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 2797
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 3111
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 3093
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 3049
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 3132
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 2643
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 3244
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2610
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2214
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2589

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177162, upper bound: 0.0176653
time: 22.24 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177162, upper bound: 0.0176829
time: 5.50 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.3561277, -3.7037566, -4.3510342, -3.7048848, -0.3918735, 0.3885664
1: -4.7724352, -3.9240489, -4.7656102, -3.9253442, -0.2474474, 0.2452656
2: -0.4650995, -0.3606079, -0.4650915, -0.3606235, -0.0156344, 0.0156419
3: -0.3913899, -0.1226121, -0.3910920, -0.1239729, -0.0601744, 0.0603970
4: -0.2425936, 0.0104689, -0.2425606, 0.0101911, -0.0428809, 0.0425938
5: -0.0895899, 0.1832755, -0.0892494, 0.1817406, -0.0557713, 0.0562285
6: -1.3818295, -0.9577419, -1.3816500, -0.9587530, -0.0448544, 0.0448912
7: 0.3506131, 0.6161983, 0.3505895, 0.6162077, -0.0384854, 0.0385551
8: -5.1386089, -4.6644974, -5.1349959, -4.6652427, -0.1638624, 0.1597808
9: -5.0380564, -4.5119286, -5.0352187, -4.5125031, -0.1491344, 0.1474891

Time for backsubstitution: 5.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: B, layer: 1, pos: 2345
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 2169
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2553
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2523
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 2767
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 425
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 3468
type: B, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 3092
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 366
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 2367
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 2344
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3111
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 3093
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 3049
type: A, layer: 1, pos: 3049
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 3132
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 2643
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2610
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2214
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2589

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177283, upper bound: 0.0173685
time: 7.22 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177281, upper bound: 0.0173863
time: 22.17 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.3562641, -3.6995990, -4.3552866, -3.6998498, -0.3888394, 0.3984458
1: -4.7724352, -3.9190478, -4.7706833, -3.9194117, -0.2423452, 0.2561823
2: -0.4651015, -0.3605612, -0.4651421, -0.3605663, -0.0155804, 0.0157895
3: -0.3925196, -0.1226051, -0.3924213, -0.1228630, -0.0625233, 0.0594103
4: -0.2425955, 0.0104634, -0.2425534, 0.0101845, -0.0428790, 0.0426265
5: -0.0909749, 0.1832761, -0.0908759, 0.1831020, -0.0586306, 0.0549235
6: -1.3821887, -0.9577414, -1.3820837, -0.9584062, -0.0457049, 0.0445048
7: 0.3506019, 0.6161983, 0.3505754, 0.6162115, -0.0385644, 0.0385163
8: -5.1386089, -4.6609125, -5.1387491, -4.6609001, -0.1596694, 0.1684982
9: -5.0380564, -4.5096335, -5.0377073, -4.5097251, -0.1464721, 0.1529719

Time for backsubstitution: 5.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 2169
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2553
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 2523
type: B, layer: 1, pos: 2523
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 2767
type: A, layer: 1, pos: 2767
type: B, layer: 1, pos: 425
type: A, layer: 1, pos: 3041
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3468
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2794
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 3092
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 366
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2367
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2344
type: A, layer: 1, pos: 2797
type: B, layer: 1, pos: 2797
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3111
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 3093
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 3049
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 3132
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 3244
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2610
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2214
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2589

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177106, upper bound: 0.0175640
time: 39.16 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177282, upper bound: 0.0175658
time: 3.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.3575039, -3.7022257, -4.3521690, -3.7048848, -0.3920194, 0.3912264
1: -4.7731113, -3.9232862, -4.7661638, -3.9253442, -0.2475199, 0.2466880
2: -0.4652393, -0.3604479, -0.4652065, -0.3606235, -0.0156512, 0.0159771
3: -0.3927606, -0.1213891, -0.3910920, -0.1229424, -0.0626244, 0.0605291
4: -0.2429202, 0.0107475, -0.2425606, 0.0104184, -0.0436445, 0.0426248
5: -0.0912940, 0.1848019, -0.0892494, 0.1829851, -0.0587186, 0.0563933
6: -1.3819773, -0.9576505, -1.3816500, -0.9586660, -0.0457802, 0.0449034
7: 0.3504539, 0.6163771, 0.3504571, 0.6162078, -0.0385032, 0.0388702
8: -5.1387033, -4.6644135, -5.1349959, -4.6651731, -0.1640238, 0.1597898
9: -5.0380497, -4.5119448, -5.0352187, -4.5125184, -0.1492663, 0.1474922

Time for backsubstitution: 5.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2346
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: B, layer: 1, pos: 2345
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 2169
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2553
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2523
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 2767
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 425
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 3468
type: B, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 3092
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 366
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 2367
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 2344
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3111
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 3093
type: B, layer: 1, pos: 3093
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 3049
type: A, layer: 1, pos: 3049
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 3132
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 2643
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2610
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2214
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2589

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177283, upper bound: 0.0175318
time: 18.07 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177282, upper bound: 0.0175504
time: 13.77 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.3576398, -3.6980686, -4.3564205, -3.6998498, -0.3889854, 0.4011062
1: -4.7731113, -3.9182837, -4.7712369, -3.9194117, -0.2424177, 0.2576046
2: -0.4652413, -0.3604012, -0.4652572, -0.3605663, -0.0155972, 0.0161247
3: -0.3938903, -0.1213820, -0.3924213, -0.1218324, -0.0649733, 0.0595424
4: -0.2429221, 0.0107419, -0.2425532, 0.0104117, -0.0436427, 0.0426575
5: -0.0926790, 0.1848026, -0.0908759, 0.1843464, -0.0615779, 0.0550883
6: -1.3823364, -0.9576499, -1.3820837, -0.9583192, -0.0466308, 0.0445170
7: 0.3504427, 0.6163771, 0.3504431, 0.6162117, -0.0385822, 0.0388315
8: -5.1387029, -4.6608291, -5.1387491, -4.6608300, -0.1598308, 0.1685072
9: -5.0380492, -4.5096507, -5.0377073, -4.5097394, -0.1466039, 0.1529749

Time for backsubstitution: 5.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 2346
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2345
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 347
type: A, layer: 1, pos: 347
type: B, layer: 1, pos: 2169
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2334
type: B, layer: 1, pos: 2334
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 2137
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2553
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 332
type: B, layer: 1, pos: 332
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 2523
type: B, layer: 1, pos: 2523
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 2767
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 3041
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3468
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 2794
type: B, layer: 1, pos: 2794
type: A, layer: 1, pos: 2851
type: B, layer: 1, pos: 2851
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 3092
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 366
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 2682
type: B, layer: 1, pos: 2682
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2367
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3511
type: B, layer: 1, pos: 3511
type: A, layer: 1, pos: 3518
type: B, layer: 1, pos: 3518
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2344
type: A, layer: 1, pos: 2797
type: B, layer: 1, pos: 2797
type: A, layer: 1, pos: 2344
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3111
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 3093
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 3049
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 3304
type: B, layer: 1, pos: 3304
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 2886
type: B, layer: 1, pos: 2886
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3482
type: B, layer: 1, pos: 3482
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 3132
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 3244
type: B, layer: 1, pos: 3244
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 2658
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 2835
type: B, layer: 1, pos: 2835
type: A, layer: 1, pos: 2836
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2836
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2610
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2214
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 2251
type: B, layer: 1, pos: 675
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 2589

## Relational analysis of NS_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177108, upper bound: 0.0175495
time: 35.29 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177281, upper bound: 0.0177295
time: 24.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 65.31 seconds
NS_A1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 65.31
Output dim: 5, lower bound: -0.0176114, upper bound: 0.0177312
NS_A1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 65.31
Output dim: 5, lower bound: -0.0176393, upper bound: 0.0177285
NS_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 65.31
Output dim: 5, lower bound: -0.0177162, upper bound: 0.0176653
NS_A2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 65.31
Output dim: 5, lower bound: -0.0177162, upper bound: 0.0176829
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 65.31
Output dim: 5, lower bound: -0.0177283, upper bound: 0.0173685
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 65.31
Output dim: 5, lower bound: -0.0177281, upper bound: 0.0173863
NS_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 65.31
Output dim: 5, lower bound: -0.0177106, upper bound: 0.0175640
NS_A2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 65.31
Output dim: 5, lower bound: -0.0177282, upper bound: 0.0175658
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 65.31
Output dim: 5, lower bound: -0.0177283, upper bound: 0.0175318
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 65.31
Output dim: 5, lower bound: -0.0177282, upper bound: 0.0175504
NS_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 65.31
Output dim: 5, lower bound: -0.0177108, upper bound: 0.0175495
NS_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 65.31
Output dim: 5, lower bound: -0.0177281, upper bound: 0.0177295

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 280.35 + 1021.80 = 1302.15 seconds
