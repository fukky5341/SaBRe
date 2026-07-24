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
execution time: IAR + RelationalAnalysis = 5.62 + 451.53 = 457.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1276895, upper bound: 0.1276922

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 892

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2639

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276870, upper bound: 0.1276969
time: 16.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276870, upper bound: 0.1276957
time: 292.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 309.15 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 309.15
Output dim: 8, lower bound: -0.1276870, upper bound: 0.1276969
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 309.15
Output dim: 8, lower bound: -0.1276870, upper bound: 0.1276957

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810238, 0.2810238
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201857, 0.4201857
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9402038, 0.9402038
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3280172, 1.3280175
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6059348, 0.6059347
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4206862, 1.4206861
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7451348, 1.7451348
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8280121, 0.8280121
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312646, 0.6312646
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351963, 0.8351963

Time for backsubstitution: 4.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2375

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276732, upper bound: 0.1276763
time: 282.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276719, upper bound: 0.1276773
time: 508.12 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810238, 0.2810238
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201857, 0.4201857
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9402038, 0.9402038
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3280172, 1.3280175
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6059348, 0.6059347
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4206862, 1.4206861
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7451348, 1.7451348
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8280121, 0.8280121
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312646, 0.6312646
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351963, 0.8351963

Time for backsubstitution: 4.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2375

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276840, upper bound: 0.1276759
time: 271.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276718, upper bound: 0.1276849
time: 699.20 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 974.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 974.78
Output dim: 8, lower bound: -0.1276732, upper bound: 0.1276763
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 974.78
Output dim: 8, lower bound: -0.1276719, upper bound: 0.1276773
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 974.78
Output dim: 8, lower bound: -0.1276840, upper bound: 0.1276759
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 974.78
Output dim: 8, lower bound: -0.1276718, upper bound: 0.1276849

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810253, 0.2810236
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201849, 0.4201998
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9402024, 0.9402032
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3280172, 1.3280175
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6059344, 0.6059344
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4206898, 1.4206861
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7451324, 1.7451526
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8280288, 0.8280115
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312646, 0.6312660
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351963, 0.8351982

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2640

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2592

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276706, upper bound: 0.1276739
time: 194.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276669, upper bound: 0.1276768
time: 34.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810236, 0.2810238
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201857, 0.4201849
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9402033, 0.9402038
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3280172, 1.3280175
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6059348, 0.6059344
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4206860, 1.4206861
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7451348, 1.7451321
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8280115, 0.8280121
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312646, 0.6312648
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351963, 0.8351963

Time for backsubstitution: 4.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 2147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2158

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276713, upper bound: 0.1276768
time: 288.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276702, upper bound: 0.1276759
time: 21.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2809664, 0.2809646
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4192365, 0.4193048
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401654, 0.9401851
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3272843, 1.3273340
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6051968, 0.6052942
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4195623, 1.4196373
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7451105, 1.7451124
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8268453, 0.8270038
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312513, 0.6312309
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351315, 0.8351478

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 3470

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 854

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276826, upper bound: 0.1276756
time: 582.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276826, upper bound: 0.1276761
time: 17.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2809646, 0.2809664
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4193048, 0.4192365
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401851, 0.9401655
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3273339, 1.3272846
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6052942, 0.6051968
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4196372, 1.4195622
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7451124, 1.7451108
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8270038, 0.8268453
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312311, 0.6312513
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351477, 0.8351316

Time for backsubstitution: 4.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3512

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276689, upper bound: 0.1276238
time: 207.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276277, upper bound: 0.1276827
time: 270.06 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 481.91 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 481.91
Output dim: 8, lower bound: -0.1276706, upper bound: 0.1276739
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 481.91
Output dim: 8, lower bound: -0.1276669, upper bound: 0.1276768
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 481.91
Output dim: 8, lower bound: -0.1276713, upper bound: 0.1276768
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 481.91
Output dim: 8, lower bound: -0.1276702, upper bound: 0.1276759
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 481.91
Output dim: 8, lower bound: -0.1276826, upper bound: 0.1276756
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 481.91
Output dim: 8, lower bound: -0.1276826, upper bound: 0.1276761
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 481.91
Output dim: 8, lower bound: -0.1276689, upper bound: 0.1276238
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 481.91
Output dim: 8, lower bound: -0.1276277, upper bound: 0.1276827

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810018, 0.2809984
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201658, 0.4201807
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401408, 0.9401425
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3271978, 1.3271974
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6057534, 0.6057380
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4198804, 1.4198767
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7448285, 1.7448446
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8270364, 0.8270193
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312578, 0.6312591
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351939, 0.8351957

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2121

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276629, upper bound: 0.1276722
time: 404.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276629, upper bound: 0.1276737
time: 185.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810000, 0.2810002
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201658, 0.4201806
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401416, 0.9401419
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3271973, 1.3271977
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6057380, 0.6057534
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4198809, 1.4198762
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7448242, 1.7448486
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8270366, 0.8270192
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312578, 0.6312591
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351938, 0.8351958

Time for backsubstitution: 4.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2567

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 841

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276498, upper bound: 0.1276577
time: 165.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276548, upper bound: 0.1276621
time: 235.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810106, 0.2810098
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201788, 0.4201781
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401673, 0.9401664
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3273821, 1.3274428
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6057619, 0.6057566
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4200013, 1.4200783
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7450385, 1.7450489
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8270071, 0.8271092
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312639, 0.6312642
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351955, 0.8351956

Time for backsubstitution: 4.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 695

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3274

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276695, upper bound: 0.1276750
time: 152.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276691, upper bound: 0.1276736
time: 206.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810096, 0.2810108
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201788, 0.4201780
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401659, 0.9401679
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3274426, 1.3273823
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6057568, 0.6057616
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4200780, 1.4200017
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7450509, 1.7450362
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8271086, 0.8270077
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312641, 0.6312640
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351957, 0.8351955

Time for backsubstitution: 4.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2519

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3318

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276601, upper bound: 0.1276700
time: 48.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276642, upper bound: 0.1276709
time: 157.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2809664, 0.2809646
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4192365, 0.4193048
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401654, 0.9401851
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3272843, 1.3273340
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6051968, 0.6052942
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4195623, 1.4196373
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7451105, 1.7451124
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8268453, 0.8270038
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312513, 0.6312309
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351315, 0.8351478

Time for backsubstitution: 4.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 812

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2485

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276762, upper bound: 0.1276768
time: 29.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276792, upper bound: 0.1276778
time: 26.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2809664, 0.2809646
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4192365, 0.4193048
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401654, 0.9401851
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3272843, 1.3273340
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6051968, 0.6052942
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4195623, 1.4196373
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7451105, 1.7451124
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8268453, 0.8270038
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312513, 0.6312309
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351315, 0.8351478

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2624

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276765, upper bound: 0.1276809
time: 17.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276765, upper bound: 0.1276832
time: 21.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2803736, 0.2804136
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4182152, 0.4181057
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9383461, 0.9384190
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3273213, 1.3272721
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6048299, 0.6047648
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4199018, 1.4198382
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7441947, 1.7441630
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8263050, 0.8262069
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6305621, 0.6305469
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8353144, 0.8353047

Time for backsubstitution: 4.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 2249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 778

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276729, upper bound: 0.1276452
time: 24.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276719, upper bound: 0.1276389
time: 380.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2804118, 0.2803754
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4181740, 0.4181470
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9384387, 0.9383264
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3273215, 1.3272718
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6048621, 0.6047326
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4199133, 1.4198265
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7441647, 1.7441931
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8263654, 0.8261466
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6305265, 0.6305823
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8353211, 0.8352982

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 2592
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2480

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276279, upper bound: 0.1276784
time: 359.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276236, upper bound: 0.1276903
time: 44.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 408.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276629, upper bound: 0.1276722
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276629, upper bound: 0.1276737
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276498, upper bound: 0.1276577
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276548, upper bound: 0.1276621
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276695, upper bound: 0.1276750
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276691, upper bound: 0.1276736
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276601, upper bound: 0.1276700
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276642, upper bound: 0.1276709
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276762, upper bound: 0.1276768
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276792, upper bound: 0.1276778
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276765, upper bound: 0.1276809
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276765, upper bound: 0.1276832
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276729, upper bound: 0.1276452
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276719, upper bound: 0.1276389
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276279, upper bound: 0.1276784
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 408.66
Output dim: 8, lower bound: -0.1276236, upper bound: 0.1276903

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810018, 0.2809984
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201658, 0.4201807
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401408, 0.9401425
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3271978, 1.3271974
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6057534, 0.6057380
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4198804, 1.4198767
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7448285, 1.7448446
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8270364, 0.8270193
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312578, 0.6312591
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351939, 0.8351957

Time for backsubstitution: 4.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 3150
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 893

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276618, upper bound: 0.1276573
time: 15.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276471, upper bound: 0.1276671
time: 261.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810018, 0.2809984
1: -1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201658, 0.4201807
2: -1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9401408, 0.9401425
3: -3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3271978, 1.3271974
4: -3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6057534, 0.6057380
5: -4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4198804, 1.4198767
6: -5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7448285, 1.7448446
7: -6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8270364, 0.8270193
8: 0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312578, 0.6312591
9: -1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351939, 0.8351957

Time for backsubstitution: 4.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2504
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2375
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3154
type: RSZ, layer: 1, pos: 3248
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2685
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 2649
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2660
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3092
type: RSZ, layer: 1, pos: 3512
type: RSZ, layer: 1, pos: 3472
type: RSZ, layer: 1, pos: 3318
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2640
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 3103
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 3542
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 3152
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 3572
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2549
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3033
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 3573
type: RSZ, layer: 1, pos: 2655
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2659
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 3151
type: RSZ, layer: 1, pos: 2204
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 2157
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 2683
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3153
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2519
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2609
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 3339
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2417
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 3288
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2177
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2067
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2605
type: RSZ, layer: 1, pos: 3366
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2291
type: RSZ, layer: 1, pos: 3042
type: RSZ, layer: 1, pos: 2065
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2176
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 3150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2504

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276670, upper bound: 0.1276701
time: 584.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276670, upper bound: 0.1276729
time: 651.71 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 1240.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 1240.32
Output dim: 8, lower bound: -0.1276618, upper bound: 0.1276573
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 1240.32
Output dim: 8, lower bound: -0.1276471, upper bound: 0.1276671
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 1240.32
Output dim: 8, lower bound: -0.1276670, upper bound: 0.1276701
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 1240.32
Output dim: 8, lower bound: -0.1276670, upper bound: 0.1276729
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1240.32
Output dim: 8, lower bound: -0.1276498, upper bound: 0.1276577
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1240.32
Output dim: 8, lower bound: -0.1276548, upper bound: 0.1276621
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1240.32
Output dim: 8, lower bound: -0.1276695, upper bound: 0.1276750
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1240.32
Output dim: 8, lower bound: -0.1276691, upper bound: 0.1276736
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1240.32
Output dim: 8, lower bound: -0.1276601, upper bound: 0.1276700
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1240.32
Output dim: 8, lower bound: -0.1276642, upper bound: 0.1276709
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1240.32
Output dim: 8, lower bound: -0.1276762, upper bound: 0.1276768
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1240.32
Output dim: 8, lower bound: -0.1276792, upper bound: 0.1276778
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1240.32
Output dim: 8, lower bound: -0.1276765, upper bound: 0.1276809
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1240.32
Output dim: 8, lower bound: -0.1276765, upper bound: 0.1276832
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1240.32
Output dim: 8, lower bound: -0.1276729, upper bound: 0.1276452
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1240.32
Output dim: 8, lower bound: -0.1276719, upper bound: 0.1276389
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1240.32
Output dim: 8, lower bound: -0.1276279, upper bound: 0.1276784
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1240.32
Output dim: 8, lower bound: -0.1276236, upper bound: 0.1276903

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 457.15 + 7727.82 = 8184.97 seconds
