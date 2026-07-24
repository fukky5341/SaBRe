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
execution time: IAR + RelationalAnalysis = 7.91 + 98.39 = 106.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.1961265, upper bound: 0.1961318

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3587
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 395
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 364
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 3179
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 283
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3550
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 393
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 479
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3182
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3418
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3204
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3403
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 296
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 446
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3356

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1960963, upper bound: 0.1961145
time: 25.56 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1961082, upper bound: 0.1961026
time: 138.56 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 164.14 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 164.14
Output dim: 5, lower bound: -0.1960963, upper bound: 0.1961145
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 164.14
Output dim: 5, lower bound: -0.1961082, upper bound: 0.1961026

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.2016311, -3.2115192, -4.2016311, -3.2115192, -0.8385985, 0.8385943
1: -6.2342000, -4.9552798, -6.2342000, -4.9552798, -0.8391595, 0.8391391
2: -1.2338924, -0.6589673, -1.2338924, -0.6589673, -0.2104370, 0.2104356
3: -0.6833572, 0.1834395, -0.6833572, 0.1834395, -0.6628004, 0.6628003
4: -0.6074311, 0.0109497, -0.6074311, 0.0109497, -0.2439231, 0.2439228
5: -0.5298702, 0.5022178, -0.5298702, 0.5022178, -0.8946170, 0.8946168
6: -0.6123511, 0.0849714, -0.6123511, 0.0849714, -0.3432240, 0.3432244
7: -1.5261405, -0.3047922, -1.5261405, -0.3047922, -0.8505267, 0.8505264
8: -4.1452231, -3.0839956, -4.1452231, -3.0839956, -0.6409538, 0.6409529
9: -5.0128598, -3.6580279, -5.0128598, -3.6580279, -0.8231863, 0.8231783

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3418
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 446
type: DSZ, layer: 1, pos: 364
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3182
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 296
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 3587
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 3204
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 479
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3403
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 283
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 393
type: DSZ, layer: 1, pos: 3179
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 395
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3550
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3349

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 794

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1960971, upper bound: 0.1961108
time: 174.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1960971, upper bound: 0.1961107
time: 180.46 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.2016311, -3.2115192, -4.2016311, -3.2115192, -0.8385943, 0.8385986
1: -6.2342000, -4.9552798, -6.2342000, -4.9552798, -0.8391390, 0.8391596
2: -1.2338924, -0.6589673, -1.2338924, -0.6589673, -0.2104356, 0.2104370
3: -0.6833572, 0.1834395, -0.6833572, 0.1834395, -0.6628003, 0.6628003
4: -0.6074311, 0.0109497, -0.6074311, 0.0109497, -0.2439229, 0.2439231
5: -0.5298702, 0.5022178, -0.5298702, 0.5022178, -0.8946169, 0.8946170
6: -0.6123511, 0.0849714, -0.6123511, 0.0849714, -0.3432243, 0.3432240
7: -1.5261405, -0.3047922, -1.5261405, -0.3047922, -0.8505265, 0.8505266
8: -4.1452231, -3.0839956, -4.1452231, -3.0839956, -0.6409528, 0.6409537
9: -5.0128598, -3.6580279, -5.0128598, -3.6580279, -0.8231783, 0.8231864

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 3204
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 3403
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 479
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 3179
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 364
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 395
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 296
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3182
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 283
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 446
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3587
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3418
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 393
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3550
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 894

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2545

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1960939, upper bound: 0.1960648
time: 182.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1960729, upper bound: 0.1960904
time: 23.83 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 211.97 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 211.97
Output dim: 5, lower bound: -0.1960971, upper bound: 0.1961108
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 211.97
Output dim: 5, lower bound: -0.1960971, upper bound: 0.1961107
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 211.97
Output dim: 5, lower bound: -0.1960939, upper bound: 0.1960648
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 211.97
Output dim: 5, lower bound: -0.1960729, upper bound: 0.1960904

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.2016311, -3.2115192, -4.2016311, -3.2115192, -0.8385985, 0.8385943
1: -6.2342000, -4.9552798, -6.2342000, -4.9552798, -0.8391595, 0.8391391
2: -1.2338924, -0.6589673, -1.2338924, -0.6589673, -0.2104370, 0.2104356
3: -0.6833572, 0.1834395, -0.6833572, 0.1834395, -0.6628004, 0.6628003
4: -0.6074311, 0.0109497, -0.6074311, 0.0109497, -0.2439231, 0.2439228
5: -0.5298702, 0.5022178, -0.5298702, 0.5022178, -0.8946170, 0.8946168
6: -0.6123511, 0.0849714, -0.6123511, 0.0849714, -0.3432240, 0.3432244
7: -1.5261405, -0.3047922, -1.5261405, -0.3047922, -0.8505267, 0.8505264
8: -4.1452231, -3.0839956, -4.1452231, -3.0839956, -0.6409538, 0.6409529
9: -5.0128598, -3.6580279, -5.0128598, -3.6580279, -0.8231863, 0.8231783

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3179
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3550
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 296
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 3403
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 479
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 395
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3587
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 364
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 283
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3182
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3418
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 446
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 393
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 3204
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3434

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 561

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1960899, upper bound: 0.1960989
time: 17.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1960806, upper bound: 0.1961089
time: 20.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.2016311, -3.2115192, -4.2016311, -3.2115192, -0.8385985, 0.8385943
1: -6.2342000, -4.9552798, -6.2342000, -4.9552798, -0.8391595, 0.8391391
2: -1.2338924, -0.6589673, -1.2338924, -0.6589673, -0.2104370, 0.2104356
3: -0.6833572, 0.1834395, -0.6833572, 0.1834395, -0.6628004, 0.6628003
4: -0.6074311, 0.0109497, -0.6074311, 0.0109497, -0.2439231, 0.2439228
5: -0.5298702, 0.5022178, -0.5298702, 0.5022178, -0.8946170, 0.8946168
6: -0.6123511, 0.0849714, -0.6123511, 0.0849714, -0.3432240, 0.3432244
7: -1.5261405, -0.3047922, -1.5261405, -0.3047922, -0.8505267, 0.8505264
8: -4.1452231, -3.0839956, -4.1452231, -3.0839956, -0.6409538, 0.6409529
9: -5.0128598, -3.6580279, -5.0128598, -3.6580279, -0.8231863, 0.8231783

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3587
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 3418
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 3403
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 395
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3182
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 446
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3204
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3179
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 479
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3550
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 364
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 283
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 296
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 393
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3351

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 750

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1961093, upper bound: 0.1961138
time: 33.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1960969, upper bound: 0.1961103
time: 168.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.2016311, -3.2115192, -4.2016311, -3.2115192, -0.8365269, 0.8366823
1: -6.2342000, -4.9552798, -6.2342000, -4.9552798, -0.8356922, 0.8359635
2: -1.2338924, -0.6589673, -1.2338924, -0.6589673, -0.2103877, 0.2103919
3: -0.6833572, 0.1834395, -0.6833572, 0.1834395, -0.6627573, 0.6627563
4: -0.6074311, 0.0109497, -0.6074311, 0.0109497, -0.2439084, 0.2439083
5: -0.5298702, 0.5022178, -0.5298702, 0.5022178, -0.8945184, 0.8945169
6: -0.6123511, 0.0849714, -0.6123511, 0.0849714, -0.3429356, 0.3429197
7: -1.5261405, -0.3047922, -1.5261405, -0.3047922, -0.8505194, 0.8505197
8: -4.1452231, -3.0839956, -4.1452231, -3.0839956, -0.6393079, 0.6393791
9: -5.0128598, -3.6580279, -5.0128598, -3.6580279, -0.8221282, 0.8221977

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 393
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 3550
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3221
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3180
type: DSZ, layer: 1, pos: 3150
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3418
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 296
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 283
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3152
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3165
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3587
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 479
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 3419
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3204
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3182
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3219
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 364
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 2168
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 3179
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 3403
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 395
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3334
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3304
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 446
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3166
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 297
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3341
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 3519

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 478

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1960914, upper bound: 0.1960365
time: 823.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1960691, upper bound: 0.1960630
time: 135.08 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 965.46 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 965.46
Output dim: 5, lower bound: -0.1960899, upper bound: 0.1960989
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 965.46
Output dim: 5, lower bound: -0.1960806, upper bound: 0.1961089
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 965.46
Output dim: 5, lower bound: -0.1961093, upper bound: 0.1961138
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 965.46
Output dim: 5, lower bound: -0.1960969, upper bound: 0.1961103
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 965.46
Output dim: 5, lower bound: -0.1960914, upper bound: 0.1960365
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 965.46
Output dim: 5, lower bound: -0.1960691, upper bound: 0.1960630
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 965.46
Output dim: 5, lower bound: -0.1960729, upper bound: 0.1960904

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 106.30 + 1955.04 = 2061.34 seconds
