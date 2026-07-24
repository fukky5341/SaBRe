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
execution time: IAR + RelationalAnalysis = 7.06 + 30.71 = 37.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.1339555, upper bound: 0.1339591

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 305
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 315
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 282
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 264
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3190
type: DSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 897

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1339553, upper bound: 0.1339597
time: 9.95 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1339553, upper bound: 0.1339562
time: 168.47 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 178.44 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 178.44
Output dim: 7, lower bound: -0.1339553, upper bound: 0.1339597
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 178.44
Output dim: 7, lower bound: -0.1339553, upper bound: 0.1339562

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.0258975, -0.1156590, -2.0258975, -0.1156590, -1.4677517, 1.4677516
1: -1.1694292, 0.2386799, -1.1694292, 0.2386799, -1.3711104, 1.3711104
2: -2.4240048, -1.8057369, -2.4240048, -1.8057369, -0.3735908, 0.3735908
3: -2.6975875, -1.4747632, -2.6975875, -1.4747632, -0.9042386, 0.9042388
4: -3.2293782, -2.1774287, -3.2293782, -2.1774287, -0.6049697, 0.6049697
5: -3.1812463, -1.8574706, -3.1812463, -1.8574706, -1.0226120, 1.0226120
6: -4.6280847, -3.3311608, -4.6280847, -3.3311608, -0.6090428, 0.6090427
7: -0.8073260, 0.6198479, -0.8073260, 0.6198479, -1.2131186, 1.2131186
8: -1.6934905, 0.1615350, -1.6934905, 0.1615350, -1.3948250, 1.3948250
9: 0.8108366, 1.4357142, 0.8108366, 1.4357142, -0.6067300, 0.6067301

Time for backsubstitution: 5.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3190
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 315
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 264
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 282
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 305
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 688

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 839

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1339547, upper bound: 0.1339567
time: 47.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1339547, upper bound: 0.1339550
time: 60.61 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.0258975, -0.1156590, -2.0258975, -0.1156590, -1.4677517, 1.4677516
1: -1.1694292, 0.2386799, -1.1694292, 0.2386799, -1.3711104, 1.3711104
2: -2.4240048, -1.8057369, -2.4240048, -1.8057369, -0.3735908, 0.3735908
3: -2.6975875, -1.4747632, -2.6975875, -1.4747632, -0.9042386, 0.9042388
4: -3.2293782, -2.1774287, -3.2293782, -2.1774287, -0.6049697, 0.6049697
5: -3.1812463, -1.8574706, -3.1812463, -1.8574706, -1.0226120, 1.0226120
6: -4.6280847, -3.3311608, -4.6280847, -3.3311608, -0.6090428, 0.6090427
7: -0.8073260, 0.6198479, -0.8073260, 0.6198479, -1.2131186, 1.2131186
8: -1.6934905, 0.1615350, -1.6934905, 0.1615350, -1.3948250, 1.3948250
9: 0.8108366, 1.4357142, 0.8108366, 1.4357142, -0.6067300, 0.6067301

Time for backsubstitution: 5.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 3190
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 264
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 315
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 305
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 282
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 2587

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2359

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1333544, upper bound: 0.1339036
time: 375.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1339003, upper bound: 0.1333566
time: 129.92 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 511.03 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 511.03
Output dim: 7, lower bound: -0.1339547, upper bound: 0.1339567
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 511.03
Output dim: 7, lower bound: -0.1339547, upper bound: 0.1339550
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 511.03
Output dim: 7, lower bound: -0.1333544, upper bound: 0.1339036
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 511.03
Output dim: 7, lower bound: -0.1339003, upper bound: 0.1333566

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.0258975, -0.1156590, -2.0258975, -0.1156590, -1.4677517, 1.4677516
1: -1.1694292, 0.2386799, -1.1694292, 0.2386799, -1.3711104, 1.3711104
2: -2.4240048, -1.8057369, -2.4240048, -1.8057369, -0.3735908, 0.3735908
3: -2.6975875, -1.4747632, -2.6975875, -1.4747632, -0.9042386, 0.9042388
4: -3.2293782, -2.1774287, -3.2293782, -2.1774287, -0.6049697, 0.6049697
5: -3.1812463, -1.8574706, -3.1812463, -1.8574706, -1.0226120, 1.0226120
6: -4.6280847, -3.3311608, -4.6280847, -3.3311608, -0.6090428, 0.6090427
7: -0.8073260, 0.6198479, -0.8073260, 0.6198479, -1.2131186, 1.2131186
8: -1.6934905, 0.1615350, -1.6934905, 0.1615350, -1.3948250, 1.3948250
9: 0.8108366, 1.4357142, 0.8108366, 1.4357142, -0.6067300, 0.6067301

Time for backsubstitution: 5.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 315
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 282
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 3190
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 264
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 305
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 501

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2047

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1339540, upper bound: 0.1339405
time: 205.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1339374, upper bound: 0.1339561
time: 175.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.0258975, -0.1156590, -2.0258975, -0.1156590, -1.4677517, 1.4677516
1: -1.1694292, 0.2386799, -1.1694292, 0.2386799, -1.3711104, 1.3711104
2: -2.4240048, -1.8057369, -2.4240048, -1.8057369, -0.3735908, 0.3735908
3: -2.6975875, -1.4747632, -2.6975875, -1.4747632, -0.9042386, 0.9042388
4: -3.2293782, -2.1774287, -3.2293782, -2.1774287, -0.6049697, 0.6049697
5: -3.1812463, -1.8574706, -3.1812463, -1.8574706, -1.0226120, 1.0226120
6: -4.6280847, -3.3311608, -4.6280847, -3.3311608, -0.6090428, 0.6090427
7: -0.8073260, 0.6198479, -0.8073260, 0.6198479, -1.2131186, 1.2131186
8: -1.6934905, 0.1615350, -1.6934905, 0.1615350, -1.3948250, 1.3948250
9: 0.8108366, 1.4357142, 0.8108366, 1.4357142, -0.6067300, 0.6067301

Time for backsubstitution: 5.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 305
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 3190
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 315
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 282
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 264
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 861

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1338507, upper bound: 0.1339471
time: 54.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1339429, upper bound: 0.1338528
time: 153.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.0258975, -0.1156590, -2.0258975, -0.1156590, -1.4610161, 1.4607570
1: -1.1694292, 0.2386799, -1.1694292, 0.2386799, -1.3709798, 1.3709774
2: -2.4240048, -1.8057369, -2.4240048, -1.8057369, -0.3728021, 0.3728037
3: -2.6975875, -1.4747632, -2.6975875, -1.4747632, -0.9009497, 0.9010658
4: -3.2293782, -2.1774287, -3.2293782, -2.1774287, -0.6037222, 0.6037627
5: -3.1812463, -1.8574706, -3.1812463, -1.8574706, -1.0190190, 1.0191386
6: -4.6280847, -3.3311608, -4.6280847, -3.3311608, -0.6017110, 0.6019873
7: -0.8073260, 0.6198479, -0.8073260, 0.6198479, -1.2130852, 1.2130854
8: -1.6934905, 0.1615350, -1.6934905, 0.1615350, -1.3925941, 1.3925080
9: 0.8108366, 1.4357142, 0.8108366, 1.4357142, -0.6067321, 0.6067321

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 315
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 264
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 305
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 282
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 3190
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2037

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 315

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1333336, upper bound: 0.1338897
time: 195.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1333426, upper bound: 0.1338835
time: 29.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.0258975, -0.1156590, -2.0258975, -0.1156590, -1.4607571, 1.4610159
1: -1.1694292, 0.2386799, -1.1694292, 0.2386799, -1.3709774, 1.3709795
2: -2.4240048, -1.8057369, -2.4240048, -1.8057369, -0.3728037, 0.3728022
3: -2.6975875, -1.4747632, -2.6975875, -1.4747632, -0.9010658, 0.9009497
4: -3.2293782, -2.1774287, -3.2293782, -2.1774287, -0.6037627, 0.6037222
5: -3.1812463, -1.8574706, -3.1812463, -1.8574706, -1.0191385, 1.0190191
6: -4.6280847, -3.3311608, -4.6280847, -3.3311608, -0.6019873, 0.6017110
7: -0.8073260, 0.6198479, -0.8073260, 0.6198479, -1.2130854, 1.2130852
8: -1.6934905, 0.1615350, -1.6934905, 0.1615350, -1.3925080, 1.3925939
9: 0.8108366, 1.4357142, 0.8108366, 1.4357142, -0.6067321, 0.6067321

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 282
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 305
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 315
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 264
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3190
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 858

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1338509, upper bound: 0.1333204
time: 28.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1338619, upper bound: 0.1333036
time: 75.89 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 110.86 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 110.86
Output dim: 7, lower bound: -0.1339540, upper bound: 0.1339405
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 110.86
Output dim: 7, lower bound: -0.1339374, upper bound: 0.1339561
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 110.86
Output dim: 7, lower bound: -0.1338507, upper bound: 0.1339471
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 110.86
Output dim: 7, lower bound: -0.1339429, upper bound: 0.1338528
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 110.86
Output dim: 7, lower bound: -0.1333336, upper bound: 0.1338897
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 110.86
Output dim: 7, lower bound: -0.1333426, upper bound: 0.1338835
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 110.86
Output dim: 7, lower bound: -0.1338509, upper bound: 0.1333204
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 110.86
Output dim: 7, lower bound: -0.1338619, upper bound: 0.1333036

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.0258975, -0.1156590, -2.0258975, -0.1156590, -1.4677438, 1.4677430
1: -1.1694292, 0.2386799, -1.1694292, 0.2386799, -1.3711044, 1.3711047
2: -2.4240048, -1.8057369, -2.4240048, -1.8057369, -0.3735150, 0.3735176
3: -2.6975875, -1.4747632, -2.6975875, -1.4747632, -0.9041914, 0.9041942
4: -3.2293782, -2.1774287, -3.2293782, -2.1774287, -0.6048236, 0.6048240
5: -3.1812463, -1.8574706, -3.1812463, -1.8574706, -1.0225768, 1.0225797
6: -4.6280847, -3.3311608, -4.6280847, -3.3311608, -0.6089200, 0.6089268
7: -0.8073260, 0.6198479, -0.8073260, 0.6198479, -1.2130742, 1.2130737
8: -1.6934905, 0.1615350, -1.6934905, 0.1615350, -1.3948271, 1.3948261
9: 0.8108366, 1.4357142, 0.8108366, 1.4357142, -0.6067145, 0.6067144

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 315
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 3190
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 264
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 282
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 305
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2318

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1338475, upper bound: 0.1338293
time: 108.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1338475, upper bound: 0.1338305
time: 119.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.0258975, -0.1156590, -2.0258975, -0.1156590, -1.4677432, 1.4677435
1: -1.1694292, 0.2386799, -1.1694292, 0.2386799, -1.3711047, 1.3711045
2: -2.4240048, -1.8057369, -2.4240048, -1.8057369, -0.3735176, 0.3735150
3: -2.6975875, -1.4747632, -2.6975875, -1.4747632, -0.9041942, 0.9041913
4: -3.2293782, -2.1774287, -3.2293782, -2.1774287, -0.6048241, 0.6048235
5: -3.1812463, -1.8574706, -3.1812463, -1.8574706, -1.0225796, 1.0225769
6: -4.6280847, -3.3311608, -4.6280847, -3.3311608, -0.6089269, 0.6089200
7: -0.8073260, 0.6198479, -0.8073260, 0.6198479, -1.2130737, 1.2130742
8: -1.6934905, 0.1615350, -1.6934905, 0.1615350, -1.3948262, 1.3948270
9: 0.8108366, 1.4357142, 0.8108366, 1.4357142, -0.6067145, 0.6067144

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 315
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3190
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 264
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 282
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 305
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 3517

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 539

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1339370, upper bound: 0.1339562
time: 33.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1339370, upper bound: 0.1339555
time: 19.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.0258975, -0.1156590, -2.0258975, -0.1156590, -1.4676104, 1.4676501
1: -1.1694292, 0.2386799, -1.1694292, 0.2386799, -1.3710160, 1.3709908
2: -2.4240048, -1.8057369, -2.4240048, -1.8057369, -0.3729379, 0.3730557
3: -2.6975875, -1.4747632, -2.6975875, -1.4747632, -0.9041860, 0.9041996
4: -3.2293782, -2.1774287, -3.2293782, -2.1774287, -0.6039333, 0.6041116
5: -3.1812463, -1.8574706, -3.1812463, -1.8574706, -1.0225159, 1.0225443
6: -4.6280847, -3.3311608, -4.6280847, -3.3311608, -0.6090391, 0.6090389
7: -0.8073260, 0.6198479, -0.8073260, 0.6198479, -1.2129407, 1.2129840
8: -1.6934905, 0.1615350, -1.6934905, 0.1615350, -1.3947108, 1.3946822
9: 0.8108366, 1.4357142, 0.8108366, 1.4357142, -0.6067291, 0.6067298

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3190
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 264
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 305
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 315
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 282
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 849

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2528

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1337986, upper bound: 0.1339146
time: 11.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1338195, upper bound: 0.1338921
time: 164.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.0258975, -0.1156590, -2.0258975, -0.1156590, -1.4676501, 1.4676102
1: -1.1694292, 0.2386799, -1.1694292, 0.2386799, -1.3709908, 1.3710160
2: -2.4240048, -1.8057369, -2.4240048, -1.8057369, -0.3730557, 0.3729379
3: -2.6975875, -1.4747632, -2.6975875, -1.4747632, -0.9041997, 0.9041859
4: -3.2293782, -2.1774287, -3.2293782, -2.1774287, -0.6041116, 0.6039333
5: -3.1812463, -1.8574706, -3.1812463, -1.8574706, -1.0225443, 1.0225159
6: -4.6280847, -3.3311608, -4.6280847, -3.3311608, -0.6090389, 0.6090391
7: -0.8073260, 0.6198479, -0.8073260, 0.6198479, -1.2129838, 1.2129406
8: -1.6934905, 0.1615350, -1.6934905, 0.1615350, -1.3946824, 1.3947105
9: 0.8108366, 1.4357142, 0.8108366, 1.4357142, -0.6067298, 0.6067291

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3190
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 315
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 305
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 282
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 264
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 2371

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2476

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1339361, upper bound: 0.1338482
time: 11.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1339391, upper bound: 0.1338437
time: 433.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.0258975, -0.1156590, -2.0258975, -0.1156590, -1.4574010, 1.4569752
1: -1.1694292, 0.2386799, -1.1694292, 0.2386799, -1.3708398, 1.3708302
2: -2.4240048, -1.8057369, -2.4240048, -1.8057369, -0.3726942, 0.3726947
3: -2.6975875, -1.4747632, -2.6975875, -1.4747632, -0.9009949, 0.9010595
4: -3.2293782, -2.1774287, -3.2293782, -2.1774287, -0.6012108, 0.6011990
5: -3.1812463, -1.8574706, -3.1812463, -1.8574706, -1.0196221, 1.0196805
6: -4.6280847, -3.3311608, -4.6280847, -3.3311608, -0.5982569, 0.5983745
7: -0.8073260, 0.6198479, -0.8073260, 0.6198479, -1.2130005, 1.2130175
8: -1.6934905, 0.1615350, -1.6934905, 0.1615350, -1.3890398, 1.3889256
9: 0.8108366, 1.4357142, 0.8108366, 1.4357142, -0.6066324, 0.6066419

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3190
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 305
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 264
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 282
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2615

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3173

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1333312, upper bound: 0.1338900
time: 84.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1333339, upper bound: 0.1333396
time: 361.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.0258975, -0.1156590, -2.0258975, -0.1156590, -1.4572339, 1.4571424
1: -1.1694292, 0.2386799, -1.1694292, 0.2386799, -1.3708324, 1.3708376
2: -2.4240048, -1.8057369, -2.4240048, -1.8057369, -0.3726932, 0.3726957
3: -2.6975875, -1.4747632, -2.6975875, -1.4747632, -0.9009434, 0.9011111
4: -3.2293782, -2.1774287, -3.2293782, -2.1774287, -0.6011585, 0.6012513
5: -3.1812463, -1.8574706, -3.1812463, -1.8574706, -1.0195612, 1.0197414
6: -4.6280847, -3.3311608, -4.6280847, -3.3311608, -0.5980983, 0.5985332
7: -0.8073260, 0.6198479, -0.8073260, 0.6198479, -1.2130172, 1.2130008
8: -1.6934905, 0.1615350, -1.6934905, 0.1615350, -1.3890114, 1.3889537
9: 0.8108366, 1.4357142, 0.8108366, 1.4357142, -0.6066419, 0.6066324

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 264
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3190
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 282
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 305
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 501

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 264

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1333208, upper bound: 0.1338467
time: 276.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1333065, upper bound: 0.1338617
time: 9.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.0258975, -0.1156590, -2.0258975, -0.1156590, -1.4606646, 1.4609475
1: -1.1694292, 0.2386799, -1.1694292, 0.2386799, -1.3708068, 1.3707888
2: -2.4240048, -1.8057369, -2.4240048, -1.8057369, -0.3721121, 0.3721452
3: -2.6975875, -1.4747632, -2.6975875, -1.4747632, -0.9009960, 0.9008806
4: -3.2293782, -2.1774287, -3.2293782, -2.1774287, -0.6032216, 0.6031888
5: -3.1812463, -1.8574706, -3.1812463, -1.8574706, -1.0189648, 1.0188640
6: -4.6280847, -3.3311608, -4.6280847, -3.3311608, -0.6019835, 0.6017073
7: -0.8073260, 0.6198479, -0.8073260, 0.6198479, -1.2129152, 1.2129555
8: -1.6934905, 0.1615350, -1.6934905, 0.1615350, -1.3923016, 1.3923621
9: 0.8108366, 1.4357142, 0.8108366, 1.4357142, -0.6067315, 0.6067316

Time for backsubstitution: 5.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3190
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 315
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 305
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 264
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 282
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 687

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3049

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1334680, upper bound: 0.1329241
time: 189.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1334557, upper bound: 0.1329370
time: 146.24 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 341.90 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 341.90
Output dim: 7, lower bound: -0.1338475, upper bound: 0.1338293
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 341.90
Output dim: 7, lower bound: -0.1338475, upper bound: 0.1338305
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 341.90
Output dim: 7, lower bound: -0.1339370, upper bound: 0.1339562
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 341.90
Output dim: 7, lower bound: -0.1339370, upper bound: 0.1339555
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 341.90
Output dim: 7, lower bound: -0.1337986, upper bound: 0.1339146
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 341.90
Output dim: 7, lower bound: -0.1338195, upper bound: 0.1338921
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 341.90
Output dim: 7, lower bound: -0.1339361, upper bound: 0.1338482
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 341.90
Output dim: 7, lower bound: -0.1339391, upper bound: 0.1338437
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 341.90
Output dim: 7, lower bound: -0.1333312, upper bound: 0.1338900
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 341.90
Output dim: 7, lower bound: -0.1333339, upper bound: 0.1333396
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 341.90
Output dim: 7, lower bound: -0.1333208, upper bound: 0.1338467
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 341.90
Output dim: 7, lower bound: -0.1333065, upper bound: 0.1338617
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 341.90
Output dim: 7, lower bound: -0.1334680, upper bound: 0.1329241
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 341.90
Output dim: 7, lower bound: -0.1334557, upper bound: 0.1329370
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 341.90
Output dim: 7, lower bound: -0.1338619, upper bound: 0.1333036

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 37.77 + 3757.15 = 3794.92 seconds
