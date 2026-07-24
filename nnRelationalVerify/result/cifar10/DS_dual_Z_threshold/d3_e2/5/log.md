## Execution arguments:
Dataset: Dataset.CIFAR10
Network: onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 5)
Time budget: 7200 seconds
Split limit: 100
Threshold: 0.12740707405


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810238, 0.2810238)
1: (-1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201857, 0.4201857)
2: (-1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9402038, 0.9402038)
3: (-3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3280172, 1.3280175)
4: (-3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6059348, 0.6059347)
5: (-4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4206862, 1.4206861)
6: (-5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7451348, 1.7451348)
7: (-6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8280121, 0.8280121)
8: (0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312646, 0.6312646)
9: (-1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351963, 0.8351963)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 5.50 + 463.73 = 469.23 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1276895, upper bound: 0.1276922

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 3573

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 3470

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1275098, upper bound: 0.1276910
time: 173.45 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276857, upper bound: 0.1275175
time: 102.82 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 276.40 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 276.40
Output dim: 8, lower bound: -0.1275098, upper bound: 0.1276910
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 276.40
Output dim: 8, lower bound: -0.1276857, upper bound: 0.1275175

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810231, 0.2810231
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201891, 0.4201891
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401696, 0.9401701
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3280323, 1.3280349
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6059027, 0.6059033
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4206810, 1.4206831
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7451444, 1.7451451
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8279482, 0.8279511
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312660, 0.6312659
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351974, 0.8351976

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 3573

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 3472

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1272207, upper bound: 0.1274110
time: 261.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1272261, upper bound: 0.1274137
time: 25.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810231, 0.2810231
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201890, 0.4201892
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401702, 0.9401696
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3280351, 1.3280324
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6059034, 0.6059027
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4206829, 1.4206808
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7451453, 1.7451446
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8279511, 0.8279482
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312660, 0.6312658
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351976, 0.8351976

Time for backsubstitution: 4.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 3573

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3472

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1274020, upper bound: 0.1272308
time: 30.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1274079, upper bound: 0.1272266
time: 334.18 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 368.49 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 368.49
Output dim: 8, lower bound: -0.1272207, upper bound: 0.1274110
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 368.49
Output dim: 8, lower bound: -0.1272261, upper bound: 0.1274137
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 368.49
Output dim: 8, lower bound: -0.1274020, upper bound: 0.1272308
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 368.49
Output dim: 8, lower bound: -0.1274079, upper bound: 0.1272266

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810231, 0.2810231
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201892, 0.4201887
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401695, 0.9401701
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3280299, 1.3280346
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6059012, 0.6059033
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4206779, 1.4206831
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7451439, 1.7451448
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8279457, 0.8279511
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312655, 0.6312659
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351972, 0.8351976

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 3573

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1271670, upper bound: 0.1274115
time: 373.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1272248, upper bound: 0.1273640
time: 14.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810231, 0.2810231
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201891, 0.4201890
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401696, 0.9401701
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3280323, 1.3280349
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6059027, 0.6059033
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4206808, 1.4206831
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7451444, 1.7451451
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8279481, 0.8279511
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312660, 0.6312659
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351974, 0.8351976

Time for backsubstitution: 3.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 3573

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 2112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1271711, upper bound: 0.1274078
time: 164.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1272251, upper bound: 0.1273490
time: 499.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810231, 0.2810231
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201890, 0.4201892
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401700, 0.9401696
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3280346, 1.3280324
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6059033, 0.6059027
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4206831, 1.4206808
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7451448, 1.7451446
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8279511, 0.8279482
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312660, 0.6312658
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351974, 0.8351976

Time for backsubstitution: 3.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 3573

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 2112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1273557, upper bound: 0.1272284
time: 167.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1274042, upper bound: 0.1271761
time: 32.18 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 203.69 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 203.69
Output dim: 8, lower bound: -0.1271670, upper bound: 0.1274115
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 203.69
Output dim: 8, lower bound: -0.1272248, upper bound: 0.1273640
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 203.69
Output dim: 8, lower bound: -0.1271711, upper bound: 0.1274078
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 203.69
Output dim: 8, lower bound: -0.1272251, upper bound: 0.1273490
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 203.69
Output dim: 8, lower bound: -0.1273557, upper bound: 0.1272284
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 203.69
Output dim: 8, lower bound: -0.1274042, upper bound: 0.1271761

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2809122, 0.2809130
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4194916, 0.4195198
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401368, 0.9401365
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3275576, 1.3275958
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6053897, 0.6053795
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4202292, 1.4202660
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7443583, 1.7444133
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8273177, 0.8273477
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6311414, 0.6311469
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351095, 0.8351154

Time for backsubstitution: 4.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 3573

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2113

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1271529, upper bound: 0.1274134
time: 41.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1271637, upper bound: 0.1273781
time: 253.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2809122, 0.2809131
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4194916, 0.4195201
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401370, 0.9401366
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3275599, 1.3275958
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6053911, 0.6053796
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4202321, 1.4202659
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7443588, 1.7444133
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8273199, 0.8273476
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6311417, 0.6311470
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351098, 0.8351156

Time for backsubstitution: 4.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 3573

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2113

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1271537, upper bound: 0.1274048
time: 562.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1271637, upper bound: 0.1273689
time: 327.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 894.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 894.21
Output dim: 8, lower bound: -0.1271529, upper bound: 0.1274134
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 894.21
Output dim: 8, lower bound: -0.1271637, upper bound: 0.1273781
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 894.21
Output dim: 8, lower bound: -0.1271537, upper bound: 0.1274048
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 894.21
Output dim: 8, lower bound: -0.1271637, upper bound: 0.1273689

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2808606, 0.2808600
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4190600, 0.4191130
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9400916, 0.9400904
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3273342, 1.3273965
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6050129, 0.6049820
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4200034, 1.4200637
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7439809, 1.7440777
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8269732, 0.8270208
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6310580, 0.6310677
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8350638, 0.8350745

Time for backsubstitution: 4.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 3573

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2592

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1271447, upper bound: 0.1274069
time: 510.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1271451, upper bound: 0.1274103
time: 23.63 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 538.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 538.64
Output dim: 8, lower bound: -0.1271447, upper bound: 0.1274069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 538.64
Output dim: 8, lower bound: -0.1271451, upper bound: 0.1274103

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2808354, 0.2808366
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4190409, 0.4190938
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9400308, 0.9400289
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3265140, 1.3265766
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6048166, 0.6048011
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4191942, 1.4192541
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7436724, 1.7437739
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8259813, 0.8260286
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6310512, 0.6310609
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8350614, 0.8350721

Time for backsubstitution: 4.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 3573

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2143

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1271383, upper bound: 0.1274066
time: 290.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1271454, upper bound: 0.1274031
time: 17.43 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 312.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 312.60
Output dim: 8, lower bound: -0.1271383, upper bound: 0.1274066
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 312.60
Output dim: 8, lower bound: -0.1271454, upper bound: 0.1274031

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 469.23 + 4244.32 = 4713.55 seconds
