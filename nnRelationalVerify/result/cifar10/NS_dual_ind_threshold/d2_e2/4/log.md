## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.1338215445


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.0258975, -0.1156590, -2.0258975, -0.1156590, -1.4677517, 1.4677516)
1: (-1.1694292, 0.2386799, -1.1694292, 0.2386799, -1.3711104, 1.3711104)
2: (-2.4240048, -1.8057369, -2.4240048, -1.8057369, -0.3735908, 0.3735908)
3: (-2.6975875, -1.4747632, -2.6975875, -1.4747632, -0.9042386, 0.9042388)
4: (-3.2293782, -2.1774287, -3.2293782, -2.1774287, -0.6049697, 0.6049697)
5: (-3.1812463, -1.8574706, -3.1812463, -1.8574706, -1.0226120, 1.0226120)
6: (-4.6280847, -3.3311608, -4.6280847, -3.3311608, -0.6090428, 0.6090427)
7: (-0.8073260, 0.6198479, -0.8073260, 0.6198479, -1.2131186, 1.2131186)
8: (-1.6934905, 0.1615350, -1.6934905, 0.1615350, -1.3948250, 1.3948250)
9: (0.8108366, 1.4357142, 0.8108366, 1.4357142, -0.6067300, 0.6067301)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.98 + 31.28 = 39.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.1339555, upper bound: 0.1339591

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 3095
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 3035
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 303
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3517
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 3026
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 3414
type: A, layer: 1, pos: 301
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 2546
type: A, layer: 1, pos: 2561
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 305
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 3230
type: A, layer: 1, pos: 248
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 3190
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 264
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 282
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 315
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3217
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 2321
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 331
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 3258
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 2501
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2276
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2277
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2256
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 3173
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2594
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2954
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3029

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3049

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1338305, upper bound: 0.1335645
time: 151.55 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1335735, upper bound: 0.1335761
time: 203.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 355.21 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 355.21
Output dim: 7, lower bound: -0.1338305, upper bound: 0.1335645
NS_A2, status: Status.VERIFIED, split count: 1, time: 355.21
Output dim: 7, lower bound: -0.1335735, upper bound: 0.1335761

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2.0258102, -0.1178818, -2.0258179, -0.1178081, -1.4648256, 1.4646788
1: -1.1693363, 0.2366126, -1.1693459, 0.2366831, -1.3691361, 1.3690929
2: -2.4237421, -1.8057373, -2.4237809, -1.8057371, -0.3730825, 0.3731641
3: -2.6948371, -1.4747987, -2.6949687, -1.4747932, -0.9005576, 0.9010217
4: -3.2285066, -2.1774302, -3.2286291, -2.1774299, -0.6040074, 0.6041570
5: -3.1780505, -1.8575158, -3.1781976, -1.8575085, -1.0184534, 1.0190086
6: -4.6273642, -3.3311608, -4.6274486, -3.3311608, -0.6072093, 0.6075350
7: -0.8060678, 0.6198460, -0.8061594, 0.6198460, -1.2117774, 1.2118310
8: -1.6934456, 0.1602345, -1.6934474, 0.1602902, -1.3929679, 1.3928032
9: 0.8108632, 1.4347584, 0.8108612, 1.4348506, -0.6059083, 0.6058358

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3095
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 3035
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 303
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3517
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 3026
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 3414
type: B, layer: 1, pos: 301
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 2546
type: B, layer: 1, pos: 2561
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 305
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 3230
type: B, layer: 1, pos: 248
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 3219
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 3190
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 264
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 282
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 315
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3217
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 263
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 2321
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 331
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 3258
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2276
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2277
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2256
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 3173
type: B, layer: 1, pos: 2278
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2594
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2954
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3029

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3095

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1335962, upper bound: 0.1332497
time: 15.58 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1335991, upper bound: 0.1333303
time: 178.01 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 199.65 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 199.65
Output dim: 7, lower bound: -0.1335962, upper bound: 0.1332497
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 199.65
Output dim: 7, lower bound: -0.1335991, upper bound: 0.1333303

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 39.27 + 554.86 = 594.12 seconds
