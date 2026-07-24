## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.1959337701


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.2016311, -3.2115192, -4.2016311, -3.2115192, -0.8386261, 0.8386261)
1: (-6.2342000, -4.9552798, -6.2342000, -4.9552798, -0.8392938, 0.8392937)
2: (-1.2338924, -0.6589673, -1.2338924, -0.6589673, -0.2104430, 0.2104430)
3: (-0.6833572, 0.1834395, -0.6833572, 0.1834395, -0.6628008, 0.6628009)
4: (-0.6074311, 0.0109497, -0.6074311, 0.0109497, -0.2439264, 0.2439264)
5: (-0.5298702, 0.5022178, -0.5298702, 0.5022178, -0.8946179, 0.8946178)
6: (-0.6123511, 0.0849714, -0.6123511, 0.0849714, -0.3432308, 0.3432308)
7: (-1.5261405, -0.3047922, -1.5261405, -0.3047922, -0.8505267, 0.8505266)
8: (-4.1452231, -3.0839956, -4.1452231, -3.0839956, -0.6409760, 0.6409761)
9: (-5.0128598, -3.6580279, -5.0128598, -3.6580279, -0.8232427, 0.8232428)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.72 + 96.80 = 104.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.1961265, upper bound: 0.1961318

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 364
type: A, layer: 1, pos: 380
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 395
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 3418
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 272
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 493
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 3181
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 3204
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 393
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 3232
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 296
type: A, layer: 1, pos: 3334
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 3520
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 2998
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3403
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3432
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3333
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 3185
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2340
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 3166
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 3165
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 3182
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 283
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 465
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 446
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3164
type: A, layer: 1, pos: 3179
type: A, layer: 1, pos: 3194
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3364
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3434
type: A, layer: 1, pos: 3587
type: A, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 364

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1961241, upper bound: 0.1959569
time: 442.30 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1961242, upper bound: 0.1961334
time: 22.22 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 464.58 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 464.58
Output dim: 5, lower bound: -0.1961241, upper bound: 0.1959569
NS_A2, status: Status.UNKNOWN, split count: 1, time: 464.58
Output dim: 5, lower bound: -0.1961242, upper bound: 0.1961334

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.2002311, -3.2111633, -4.2004642, -3.2116342, -0.8367306, 0.8372092
1: -6.2314396, -4.9551067, -6.2318377, -4.9553456, -0.8365902, 0.8373909
2: -1.2292218, -0.6616609, -1.2298069, -0.6592464, -0.2069642, 0.2043543
3: -0.6788311, 0.1768959, -0.6822884, 0.1774786, -0.6523370, 0.6552343
4: -0.6067684, 0.0104250, -0.6071603, 0.0105538, -0.2429674, 0.2430816
5: -0.5250055, 0.4938587, -0.5294647, 0.4945914, -0.8823969, 0.8858492
6: -0.6074985, 0.0842125, -0.6082228, 0.0845912, -0.3364639, 0.3367445
7: -1.5221142, -0.3089814, -1.5224752, -0.3071832, -0.8439508, 0.8428714
8: -4.1430225, -3.0861795, -4.1442747, -3.0859592, -0.6371099, 0.6379446
9: -5.0099821, -3.6577537, -5.0104384, -3.6580839, -0.8202549, 0.8212313

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 380
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 395
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3418
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 3219
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 272
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 493
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 364
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 3204
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 393
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3232
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 296
type: B, layer: 1, pos: 3334
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3520
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3403
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 3432
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3333
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 3185
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 3166
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 3208
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 3165
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 3182
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 283
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 465
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 446
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 2324
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3164
type: B, layer: 1, pos: 3179
type: B, layer: 1, pos: 3194
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3364
type: B, layer: 1, pos: 3366
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3434
type: B, layer: 1, pos: 3587
type: B, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 380

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1958765, upper bound: 0.1959308
time: 16.59 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1961197, upper bound: 0.1959565
time: 25.99 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.2011113, -3.2115293, -4.2011642, -3.2115281, -0.8380886, 0.8380722
1: -6.2335424, -4.9552965, -6.2336106, -4.9552937, -0.8388549, 0.8385668
2: -1.2338889, -0.6589695, -1.2338892, -0.6589692, -0.2090262, 0.2104201
3: -0.6833496, 0.1833856, -0.6833503, 0.1833909, -0.6627446, 0.6623008
4: -0.6074304, 0.0109470, -0.6074303, 0.0109473, -0.2438371, 0.2437660
5: -0.5298648, 0.5021647, -0.5298653, 0.5021700, -0.8945513, 0.8937696
6: -0.6121736, 0.0849710, -0.6121898, 0.0849710, -0.3431996, 0.3432218
7: -1.5261319, -0.3048131, -1.5261326, -0.3048114, -0.8498339, 0.8504992
8: -4.1452165, -3.0841789, -4.1452169, -3.0841601, -0.6406783, 0.6399634
9: -5.0124302, -3.6580405, -5.0124664, -3.6580391, -0.8226298, 0.8227323

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 380
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 395
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3418
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 3219
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 272
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 364
type: B, layer: 1, pos: 493
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 3204
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 393
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 3232
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 296
type: B, layer: 1, pos: 3334
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 3520
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3403
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 3432
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3333
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 3185
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 3166
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 3208
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 3165
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 3182
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 283
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 465
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 446
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 2324
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3164
type: B, layer: 1, pos: 3179
type: B, layer: 1, pos: 3194
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3364
type: B, layer: 1, pos: 3366
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3434
type: B, layer: 1, pos: 3587
type: B, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 380

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1958752, upper bound: 0.1960979
time: 171.79 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1961196, upper bound: 0.1961205
time: 189.10 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 366.90 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 366.90
Output dim: 5, lower bound: -0.1958765, upper bound: 0.1959308
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 366.90
Output dim: 5, lower bound: -0.1961197, upper bound: 0.1959565
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 366.90
Output dim: 5, lower bound: -0.1958752, upper bound: 0.1960979
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 366.90
Output dim: 5, lower bound: -0.1961196, upper bound: 0.1961205

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.2002292, -3.2114146, -4.2007771, -3.2118430, -0.8354351, 0.8366303
1: -6.2314034, -4.9579449, -6.2342567, -4.9584594, -0.8343010, 0.8391311
2: -1.2292186, -0.6616610, -1.2299138, -0.6585725, -0.2072083, 0.2042003
3: -0.6788185, 0.1768427, -0.6887482, 0.1776021, -0.6515489, 0.6616487
4: -0.6067213, 0.0104237, -0.6086857, 0.0115038, -0.2433981, 0.2452124
5: -0.5249857, 0.4937972, -0.5371292, 0.4949776, -0.8815306, 0.8934448
6: -0.6074502, 0.0840241, -0.6084591, 0.0856358, -0.3373550, 0.3362275
7: -1.5221007, -0.3089830, -1.5224751, -0.3004781, -0.8507012, 0.8413869
8: -4.1430206, -3.0862393, -4.1571503, -3.0860147, -0.6327109, 0.6508763
9: -5.0099349, -3.6582949, -5.0106878, -3.6586356, -0.8192737, 0.8214905

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 395
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 3418
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 272
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 380
type: A, layer: 1, pos: 493
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 3181
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 3204
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 393
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3232
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 296
type: A, layer: 1, pos: 3334
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 3520
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 2998
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3403
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 3432
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3333
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 3185
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2340
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3166
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 3165
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 3182
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 283
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 465
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 446
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3164
type: A, layer: 1, pos: 3179
type: A, layer: 1, pos: 3194
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3364
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3434
type: A, layer: 1, pos: 3587
type: A, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 386

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1960881, upper bound: 0.1957963
time: 20.44 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1961121, upper bound: 0.1959434
time: 265.67 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.2011032, -3.2118778, -4.2011552, -3.2119184, -0.8366973, 0.8367494
1: -6.2333555, -4.9569020, -6.2334027, -4.9571037, -0.8364501, 0.8364091
2: -1.2334586, -0.6589721, -1.2334054, -0.6589723, -0.2084526, 0.2098180
3: -0.6830746, 0.1806178, -0.6830419, 0.1803234, -0.6594367, 0.6592619
4: -0.6072279, 0.0109423, -0.6072028, 0.0109422, -0.2435522, 0.2434309
5: -0.5297199, 0.4990314, -0.5297046, 0.4986956, -0.8909663, 0.8905135
6: -0.6111857, 0.0846392, -0.6110784, 0.0845979, -0.3411798, 0.3410413
7: -1.5233157, -0.3049789, -1.5229595, -0.3049979, -0.8469895, 0.8473043
8: -4.1450348, -3.0886126, -4.1450138, -3.0891559, -0.6355529, 0.6353813
9: -5.0119524, -3.6583409, -5.0119314, -3.6583703, -0.8214010, 0.8214744

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 395
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 3418
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 272
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 380
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 493
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 3181
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 3204
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 393
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 3232
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 296
type: A, layer: 1, pos: 3334
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 3520
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 2998
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3403
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 3432
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3333
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 3185
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2340
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3166
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 3165
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 3182
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 283
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 465
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 446
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3164
type: A, layer: 1, pos: 3179
type: A, layer: 1, pos: 3194
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3364
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3434
type: A, layer: 1, pos: 3587
type: A, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 386

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1958462, upper bound: 0.1959380
time: 159.26 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1958677, upper bound: 0.1960904
time: 190.32 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4.2011099, -3.2117815, -4.2013807, -3.2117357, -0.8368104, 0.8373701
1: -6.2335076, -4.9581327, -6.2358828, -4.9584084, -0.8365592, 0.8403084
2: -1.2338859, -0.6589696, -1.2339945, -0.6582953, -0.2092156, 0.2100503
3: -0.6833369, 0.1833336, -0.6898037, 0.1835155, -0.6619551, 0.6687078
4: -0.6073833, 0.0109457, -0.6089528, 0.0118969, -0.2442566, 0.2458625
5: -0.5298452, 0.5021087, -0.5375268, 0.5025601, -0.8936834, 0.9013612
6: -0.6121632, 0.0847828, -0.6124693, 0.0860120, -0.3440574, 0.3427020
7: -1.5261176, -0.3048145, -1.5261302, -0.2981068, -0.8565838, 0.8490127
8: -4.1452141, -3.0842381, -4.1580920, -3.0842152, -0.6362762, 0.6529100
9: -5.0123844, -3.6585808, -5.0126595, -3.6585920, -0.8217250, 0.8229158

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 395
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 297
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 3418
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 3219
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 272
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 380
type: A, layer: 1, pos: 493
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 3181
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 3204
type: A, layer: 1, pos: 3221
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 393
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 3232
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 296
type: A, layer: 1, pos: 3334
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 3520
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 2998
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 3304
type: A, layer: 1, pos: 3403
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 3432
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3333
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 3185
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3180
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2340
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 3166
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3341
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 3165
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 3182
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 283
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 465
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 446
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3164
type: A, layer: 1, pos: 3179
type: A, layer: 1, pos: 3194
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3364
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3434
type: A, layer: 1, pos: 3587
type: A, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 386

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1960894, upper bound: 0.1959652
time: 34.13 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1961112, upper bound: 0.1961190
time: 18.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 58.36 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 58.36
Output dim: 5, lower bound: -0.1960881, upper bound: 0.1957963
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 58.36
Output dim: 5, lower bound: -0.1961121, upper bound: 0.1959434
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 58.36
Output dim: 5, lower bound: -0.1958462, upper bound: 0.1959380
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 58.36
Output dim: 5, lower bound: -0.1958677, upper bound: 0.1960904
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 58.36
Output dim: 5, lower bound: -0.1960894, upper bound: 0.1959652
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 58.36
Output dim: 5, lower bound: -0.1961112, upper bound: 0.1961190

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.1993346, -3.2127037, -4.2005963, -3.2129917, -0.8334064, 0.8351887
1: -6.2293954, -4.9602718, -6.2340927, -4.9606166, -0.8276659, 0.8355148
2: -1.2290015, -0.6621888, -1.2298566, -0.6590792, -0.2052730, 0.2030732
3: -0.6750944, 0.1729164, -0.6884215, 0.1739879, -0.6439409, 0.6572790
4: -0.6048427, 0.0087129, -0.6070287, 0.0114981, -0.2413607, 0.2414686
5: -0.5212734, 0.4899151, -0.5369494, 0.4914098, -0.8740577, 0.8893048
6: -0.6063728, 0.0834093, -0.6074831, 0.0851053, -0.3352036, 0.3341618
7: -1.5212181, -0.3098115, -1.5216755, -0.3005144, -0.8499333, 0.8403768
8: -4.1364312, -3.0929375, -4.1571345, -3.0920651, -0.6200699, 0.6442086
9: -5.0088253, -3.6585107, -5.0098968, -3.6588240, -0.8179623, 0.8202851

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 395
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 297
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3418
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 3219
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 272
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 493
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 364
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 3204
type: B, layer: 1, pos: 3221
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 393
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3232
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 296
type: B, layer: 1, pos: 3334
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3520
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 3304
type: B, layer: 1, pos: 3403
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 3432
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 3333
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 3185
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3180
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 3341
type: B, layer: 1, pos: 3166
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 3208
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 3165
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 3182
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 283
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 465
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 446
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 2324
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3164
type: B, layer: 1, pos: 3179
type: B, layer: 1, pos: 3194
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3364
type: B, layer: 1, pos: 3366
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3434
type: B, layer: 1, pos: 3587
type: B, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 354

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1958930, upper bound: 0.1957832
time: 21.74 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1960807, upper bound: 0.1957865
time: 282.15 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 104.51 + 1896.32 = 2000.83 seconds
