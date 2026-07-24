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
execution time: IAR + RelationalAnalysis = 8.27 + 30.88 = 39.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.1339555, upper bound: 0.1339591

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 264
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 315
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3190
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3230
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 282
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 305
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2276
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2954
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3173
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3517

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 306

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1337902, upper bound: 0.1337544
time: 224.69 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1337534, upper bound: 0.1337941
time: 378.57 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 603.33 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 603.33
Output dim: 7, lower bound: -0.1337902, upper bound: 0.1337544
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 603.33
Output dim: 7, lower bound: -0.1337534, upper bound: 0.1337941

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 39.15 + 603.34 = 642.48 seconds
