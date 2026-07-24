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
execution time: IAR + RelationalAnalysis = 8.18 + 266.74 = 274.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0178986, upper bound: 0.0178987

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 366
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 3518
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2493

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 366

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177551, upper bound: 0.0177424
time: 4.21 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0177405, upper bound: 0.0177556
time: 40.67 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 44.90 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 44.90
Output dim: 5, lower bound: -0.0177551, upper bound: 0.0177424
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 44.90
Output dim: 5, lower bound: -0.0177405, upper bound: 0.0177556

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 274.92 + 44.90 = 319.81 seconds
