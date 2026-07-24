## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 8)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.08591939459999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.2422794, 0.4328457, -0.2422794, 0.4328457, -0.5352136, 0.5352136)
1: (-0.2577629, 1.6339651, -0.2577629, 1.6339651, -1.4875588, 1.4875588)
2: (-1.8678656, -0.9734896, -1.8678656, -0.9734896, -0.4670140, 0.4670140)
3: (-2.5467930, -1.0992618, -2.5467930, -1.0992618, -0.5658760, 0.5658760)
4: (-3.5781426, -2.2169888, -3.5781426, -2.2169888, -0.7696934, 0.7696935)
5: (-2.6300147, -1.2237928, -2.6300147, -1.2237928, -0.5257534, 0.5257534)
6: (-7.3644829, -5.4154067, -7.3644829, -5.4154067, -0.6978738, 0.6978740)
7: (-1.9149994, -0.6417733, -1.9149994, -0.6417733, -0.6464138, 0.6464137)
8: (-0.3082114, 0.2517074, -0.3082114, 0.2517074, -0.0906395, 0.0906395)
9: (-0.8158433, 0.1592232, -0.8158433, 0.1592232, -0.7570585, 0.7570581)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.69 + 185.07 = 192.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0860054, upper bound: 0.0859961

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 277
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 617
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 632
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 673
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3560
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3595

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3061

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859842, upper bound: 0.0859872
time: 24.54 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859833, upper bound: 0.0859896
time: 238.82 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 263.44 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 263.44
Output dim: 0, lower bound: -0.0859842, upper bound: 0.0859872
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 263.44
Output dim: 0, lower bound: -0.0859833, upper bound: 0.0859896

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.2422794, 0.4328457, -0.2422794, 0.4328457, -0.5352124, 0.5352126
1: -0.2577629, 1.6339651, -0.2577629, 1.6339651, -1.4874868, 1.4874969
2: -1.8678656, -0.9734896, -1.8678656, -0.9734896, -0.4669768, 0.4669764
3: -2.5467930, -1.0992618, -2.5467930, -1.0992618, -0.5658441, 0.5658437
4: -3.5781426, -2.2169888, -3.5781426, -2.2169888, -0.7696892, 0.7696891
5: -2.6300147, -1.2237928, -2.6300147, -1.2237928, -0.5256991, 0.5256988
6: -7.3644829, -5.4154067, -7.3644829, -5.4154067, -0.6977026, 0.6977034
7: -1.9149994, -0.6417733, -1.9149994, -0.6417733, -0.6463891, 0.6463890
8: -0.3082114, 0.2517074, -0.3082114, 0.2517074, -0.0906394, 0.0906394
9: -0.8158433, 0.1592232, -0.8158433, 0.1592232, -0.7570513, 0.7570512

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 277
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 617
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 632
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 673
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3560
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3595

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3062

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859718, upper bound: 0.0859600
time: 181.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859624, upper bound: 0.0859735
time: 286.84 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.2422794, 0.4328457, -0.2422794, 0.4328457, -0.5352126, 0.5352125
1: -0.2577629, 1.6339651, -0.2577629, 1.6339651, -1.4874969, 1.4874871
2: -1.8678656, -0.9734896, -1.8678656, -0.9734896, -0.4669764, 0.4669768
3: -2.5467930, -1.0992618, -2.5467930, -1.0992618, -0.5658438, 0.5658439
4: -3.5781426, -2.2169888, -3.5781426, -2.2169888, -0.7696892, 0.7696892
5: -2.6300147, -1.2237928, -2.6300147, -1.2237928, -0.5256989, 0.5256991
6: -7.3644829, -5.4154067, -7.3644829, -5.4154067, -0.6977034, 0.6977028
7: -1.9149994, -0.6417733, -1.9149994, -0.6417733, -0.6463890, 0.6463890
8: -0.3082114, 0.2517074, -0.3082114, 0.2517074, -0.0906394, 0.0906394
9: -0.8158433, 0.1592232, -0.8158433, 0.1592232, -0.7570513, 0.7570516

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 277
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 617
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 632
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 673
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3560
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3595

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3062

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859674, upper bound: 0.0859707
time: 28.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859575, upper bound: 0.0859761
time: 19.46 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 53.93 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 53.93
Output dim: 0, lower bound: -0.0859718, upper bound: 0.0859600
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 53.93
Output dim: 0, lower bound: -0.0859624, upper bound: 0.0859735
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 53.93
Output dim: 0, lower bound: -0.0859674, upper bound: 0.0859707
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 53.93
Output dim: 0, lower bound: -0.0859575, upper bound: 0.0859761

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.2422794, 0.4328457, -0.2422794, 0.4328457, -0.5351790, 0.5351794
1: -0.2577629, 1.6339651, -0.2577629, 1.6339651, -1.4875145, 1.4875305
2: -1.8678656, -0.9734896, -1.8678656, -0.9734896, -0.4670237, 0.4670234
3: -2.5467930, -1.0992618, -2.5467930, -1.0992618, -0.5562704, 0.5563163
4: -3.5781426, -2.2169888, -3.5781426, -2.2169888, -0.7680011, 0.7680061
5: -2.6300147, -1.2237928, -2.6300147, -1.2237928, -0.5158195, 0.5158688
6: -7.3644829, -5.4154067, -7.3644829, -5.4154067, -0.6931028, 0.6931300
7: -1.9149994, -0.6417733, -1.9149994, -0.6417733, -0.6390300, 0.6390681
8: -0.3082114, 0.2517074, -0.3082114, 0.2517074, -0.0906411, 0.0906408
9: -0.8158433, 0.1592232, -0.8158433, 0.1592232, -0.7570496, 0.7570490

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 277
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 617
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 632
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 673
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3560
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3595

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859660, upper bound: 0.0859557
time: 25.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859627, upper bound: 0.0859638
time: 22.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.2422794, 0.4328457, -0.2422794, 0.4328457, -0.5351793, 0.5351790
1: -0.2577629, 1.6339651, -0.2577629, 1.6339651, -1.4875188, 1.4875243
2: -1.8678656, -0.9734896, -1.8678656, -0.9734896, -0.4670240, 0.4670220
3: -2.5467930, -1.0992618, -2.5467930, -1.0992618, -0.5563152, 0.5562700
4: -3.5781426, -2.2169888, -3.5781426, -2.2169888, -0.7680054, 0.7680010
5: -2.6300147, -1.2237928, -2.6300147, -1.2237928, -0.5158671, 0.5158194
6: -7.3644829, -5.4154067, -7.3644829, -5.4154067, -0.6931243, 0.6931037
7: -1.9149994, -0.6417733, -1.9149994, -0.6417733, -0.6390667, 0.6390299
8: -0.3082114, 0.2517074, -0.3082114, 0.2517074, -0.0906408, 0.0906411
9: -0.8158433, 0.1592232, -0.8158433, 0.1592232, -0.7570491, 0.7570491

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 277
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 617
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 632
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 673
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3560
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3595

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859575, upper bound: 0.0859626
time: 182.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859523, upper bound: 0.0859660
time: 264.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.2422794, 0.4328457, -0.2422794, 0.4328457, -0.5351790, 0.5351793
1: -0.2577629, 1.6339651, -0.2577629, 1.6339651, -1.4875240, 1.4875188
2: -1.8678656, -0.9734896, -1.8678656, -0.9734896, -0.4670221, 0.4670239
3: -2.5467930, -1.0992618, -2.5467930, -1.0992618, -0.5562701, 0.5563152
4: -3.5781426, -2.2169888, -3.5781426, -2.2169888, -0.7680008, 0.7680054
5: -2.6300147, -1.2237928, -2.6300147, -1.2237928, -0.5158194, 0.5158671
6: -7.3644829, -5.4154067, -7.3644829, -5.4154067, -0.6931038, 0.6931242
7: -1.9149994, -0.6417733, -1.9149994, -0.6417733, -0.6390299, 0.6390667
8: -0.3082114, 0.2517074, -0.3082114, 0.2517074, -0.0906411, 0.0906408
9: -0.8158433, 0.1592232, -0.8158433, 0.1592232, -0.7570491, 0.7570492

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 277
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 617
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 632
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 673
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3560
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3595

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2131

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859657, upper bound: 0.0859588
time: 25.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859590, upper bound: 0.0859630
time: 86.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.2422794, 0.4328457, -0.2422794, 0.4328457, -0.5351793, 0.5351790
1: -0.2577629, 1.6339651, -0.2577629, 1.6339651, -1.4875305, 1.4875145
2: -1.8678656, -0.9734896, -1.8678656, -0.9734896, -0.4670235, 0.4670238
3: -2.5467930, -1.0992618, -2.5467930, -1.0992618, -0.5563161, 0.5562702
4: -3.5781426, -2.2169888, -3.5781426, -2.2169888, -0.7680061, 0.7680011
5: -2.6300147, -1.2237928, -2.6300147, -1.2237928, -0.5158688, 0.5158195
6: -7.3644829, -5.4154067, -7.3644829, -5.4154067, -0.6931300, 0.6931030
7: -1.9149994, -0.6417733, -1.9149994, -0.6417733, -0.6390680, 0.6390299
8: -0.3082114, 0.2517074, -0.3082114, 0.2517074, -0.0906408, 0.0906411
9: -0.8158433, 0.1592232, -0.8158433, 0.1592232, -0.7570491, 0.7570494

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 277
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 617
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 632
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 673
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3560
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3595

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2131

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859587, upper bound: 0.0859668
time: 144.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859489, upper bound: 0.0859697
time: 13.59 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 163.68 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 163.68
Output dim: 0, lower bound: -0.0859660, upper bound: 0.0859557
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 163.68
Output dim: 0, lower bound: -0.0859627, upper bound: 0.0859638
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 163.68
Output dim: 0, lower bound: -0.0859575, upper bound: 0.0859626
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 163.68
Output dim: 0, lower bound: -0.0859523, upper bound: 0.0859660
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 163.68
Output dim: 0, lower bound: -0.0859657, upper bound: 0.0859588
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 163.68
Output dim: 0, lower bound: -0.0859590, upper bound: 0.0859630
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 163.68
Output dim: 0, lower bound: -0.0859587, upper bound: 0.0859668
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 163.68
Output dim: 0, lower bound: -0.0859489, upper bound: 0.0859697

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.2422794, 0.4328457, -0.2422794, 0.4328457, -0.5351789, 0.5351791
1: -0.2577629, 1.6339651, -0.2577629, 1.6339651, -1.4875143, 1.4875302
2: -1.8678656, -0.9734896, -1.8678656, -0.9734896, -0.4670236, 0.4670233
3: -2.5467930, -1.0992618, -2.5467930, -1.0992618, -0.5562700, 0.5563161
4: -3.5781426, -2.2169888, -3.5781426, -2.2169888, -0.7680008, 0.7680057
5: -2.6300147, -1.2237928, -2.6300147, -1.2237928, -0.5158192, 0.5158685
6: -7.3644829, -5.4154067, -7.3644829, -5.4154067, -0.6931028, 0.6931299
7: -1.9149994, -0.6417733, -1.9149994, -0.6417733, -0.6390284, 0.6390673
8: -0.3082114, 0.2517074, -0.3082114, 0.2517074, -0.0906396, 0.0906396
9: -0.8158433, 0.1592232, -0.8158433, 0.1592232, -0.7570493, 0.7570488

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 277
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 617
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 632
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 673
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2443
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 3330
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3347
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3555
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3560
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3570
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 3591
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3595

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2116

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859596, upper bound: 0.0859501
time: 16.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0859587, upper bound: 0.0859472
time: 409.11 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 431.38 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 431.38
Output dim: 0, lower bound: -0.0859596, upper bound: 0.0859501
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 431.38
Output dim: 0, lower bound: -0.0859587, upper bound: 0.0859472
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 431.38
Output dim: 0, lower bound: -0.0859627, upper bound: 0.0859638
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 431.38
Output dim: 0, lower bound: -0.0859575, upper bound: 0.0859626
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 431.38
Output dim: 0, lower bound: -0.0859523, upper bound: 0.0859660
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 431.38
Output dim: 0, lower bound: -0.0859657, upper bound: 0.0859588
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 431.38
Output dim: 0, lower bound: -0.0859590, upper bound: 0.0859630
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 431.38
Output dim: 0, lower bound: -0.0859587, upper bound: 0.0859668
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 431.38
Output dim: 0, lower bound: -0.0859489, upper bound: 0.0859697

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 192.75 + 2011.24 = 2203.99 seconds
