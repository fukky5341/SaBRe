## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 11)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.10251260279999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3428798, 1.3428802)
1: (-1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8303232, 1.8303232)
2: (-0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2348176, 0.2348175)
3: (-0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4901261, 0.4901261)
4: (-2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2651898, 0.2651898)
5: (-0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6315912, 0.6315912)
6: (-3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2626626, 0.2626626)
7: (-0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6587952, 0.6587954)
8: (-4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0975242, 1.0975242)
9: (-1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4228096, 1.4228096)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.35 + 36.04 = 44.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.1029240, upper bound: 0.1029258

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3035
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2344
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2343
type: A, layer: 1, pos: 2813
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 3107
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 3455
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3105
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2458
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2871
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2848
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 2340
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 885
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2462
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2909
type: A, layer: 1, pos: 2921
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2359

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027647, upper bound: 0.1028354
time: 14.01 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1028394, upper bound: 0.1028417
time: 8.25 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 22.33 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 22.33
Output dim: 3, lower bound: -0.1027647, upper bound: 0.1028354
NS_A2, status: Status.UNKNOWN, split count: 1, time: 22.33
Output dim: 3, lower bound: -0.1028394, upper bound: 0.1028417

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.9058614, -2.3808494, -3.9077125, -2.3800001, -1.3329201, 1.3353739
1: -1.6632314, 0.3498440, -1.6670070, 0.3511267, -1.8178759, 1.8205276
2: -0.6077659, -0.3019880, -0.6078733, -0.3018690, -0.2341944, 0.2335923
3: -0.3662307, 0.1831158, -0.3666571, 0.1840298, -0.4875154, 0.4872714
4: -2.4625673, -1.7340959, -2.4626851, -1.7337486, -0.2643761, 0.2638802
5: -0.5977601, 0.1129704, -0.5982838, 0.1139134, -0.6286067, 0.6285381
6: -3.3371434, -2.4722021, -3.3374548, -2.4709759, -0.2611068, 0.2606411
7: -0.3173715, 0.4371143, -0.3174303, 0.4358639, -0.6584446, 0.6595604
8: -4.6507998, -2.6750693, -4.6517353, -2.6745262, -1.0881002, 1.0911300
9: -1.8708081, -0.0019269, -1.8729534, -0.0011463, -1.4149795, 1.4167085

Time for backsubstitution: 6.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3035
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2343
type: B, layer: 1, pos: 2813
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3107
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 3455
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2458
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2871
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 885
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
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3035

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027376, upper bound: 0.1026549
time: 142.42 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027400, upper bound: 0.1028124
time: 14.58 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.9078131, -2.3771594, -3.9078255, -2.3767633, -1.3411374, 1.3382187
1: -1.6670499, 0.3564477, -1.6670547, 0.3569746, -1.8277731, 1.8258781
2: -0.6081595, -0.3018690, -0.6082063, -0.3018690, -0.2336267, 0.2343566
3: -0.3676799, 0.1840391, -0.3678977, 0.1840396, -0.4884270, 0.4890839
4: -2.4631751, -1.7337438, -2.4631937, -1.7337440, -0.2638463, 0.2651372
5: -0.5993920, 0.1139279, -0.5996658, 0.1139288, -0.6293784, 0.6301829
6: -3.3387599, -2.4709744, -3.3388057, -2.4709737, -0.2605763, 0.2622846
7: -0.3165116, 0.4358659, -0.3167254, 0.4358724, -0.6589090, 0.6579430
8: -4.6517725, -2.6728845, -4.6517787, -2.6725864, -1.0965450, 1.0897181
9: -1.8730154, 0.0019832, -1.8730226, 0.0023069, -1.4212742, 1.4197655

Time for backsubstitution: 6.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3035
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2343
type: B, layer: 1, pos: 2813
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3107
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 3455
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2458
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2871
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 885
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
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3035

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1028113, upper bound: 0.1026573
time: 30.27 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1028141, upper bound: 0.1028163
time: 24.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 61.28 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 61.28
Output dim: 3, lower bound: -0.1027376, upper bound: 0.1026549
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 61.28
Output dim: 3, lower bound: -0.1027400, upper bound: 0.1028124
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 61.28
Output dim: 3, lower bound: -0.1028113, upper bound: 0.1026573
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 61.28
Output dim: 3, lower bound: -0.1028141, upper bound: 0.1028163

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3.9056325, -2.3892670, -3.9053950, -2.3899674, -1.3225365, 1.3240662
1: -1.6631427, 0.3392615, -1.6643233, 0.3385105, -1.8049746, 1.8067541
2: -0.6071895, -0.3019880, -0.6071765, -0.3019015, -0.2335194, 0.2330693
3: -0.3628505, 0.1830993, -0.3626175, 0.1830321, -0.4830534, 0.4831735
4: -2.4621215, -1.7341031, -2.4621480, -1.7337120, -0.2634653, 0.2631409
5: -0.5939029, 0.1129470, -0.5936625, 0.1129896, -0.6234487, 0.6236942
6: -3.3347254, -2.4722068, -3.3344622, -2.4723663, -0.2574157, 0.2577752
7: -0.3170732, 0.4370846, -0.3170986, 0.4367608, -0.6587379, 0.6591465
8: -4.6507068, -2.6799698, -4.6513720, -2.6803226, -1.0807729, 1.0824327
9: -1.8706813, -0.0072718, -1.8721542, -0.0075378, -1.4081073, 1.4095163

Time for backsubstitution: 6.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2344
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2343
type: A, layer: 1, pos: 2813
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3107
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 3035
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 3455
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3105
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2458
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2871
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 2848
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 2340
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 885
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2462
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2909
type: A, layer: 1, pos: 2921
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3039

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025550, upper bound: 0.1026053
time: 12.84 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1026861, upper bound: 0.1026045
time: 20.66 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.9058418, -2.3813939, -3.9076891, -2.3806729, -1.3295164, 1.3349152
1: -1.6632233, 0.3488846, -1.6669965, 0.3499293, -1.8155732, 1.8197346
2: -0.6077032, -0.3019880, -0.6077956, -0.3018690, -0.2339094, 0.2332758
3: -0.3659391, 0.1831142, -0.3663129, 0.1840279, -0.4872653, 0.4863199
4: -2.4624681, -1.7340970, -2.4625602, -1.7337492, -0.2643689, 0.2628910
5: -0.5974010, 0.1129684, -0.5978425, 0.1139089, -0.6283140, 0.6273739
6: -3.3371267, -2.4722052, -3.3374338, -2.4709804, -0.2610794, 0.2566984
7: -0.3170055, 0.4371028, -0.3169663, 0.4358490, -0.6581802, 0.6594586
8: -4.6507864, -2.6756811, -4.6517200, -2.6752849, -1.0820119, 1.0909867
9: -1.8707967, -0.0026398, -1.8729382, -0.0020504, -1.4142747, 1.4162331

Time for backsubstitution: 6.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2344
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2343
type: A, layer: 1, pos: 2813
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3035
type: A, layer: 1, pos: 3107
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 3455
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3105
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2458
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2871
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2848
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 2340
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 885
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2462
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2909
type: A, layer: 1, pos: 2921
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3039

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025566, upper bound: 0.1027615
time: 16.38 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1026886, upper bound: 0.1027594
time: 158.30 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.9075947, -2.3855729, -3.9055090, -2.3867378, -1.3307462, 1.3269100
1: -1.6669598, 0.3458362, -1.6643715, 0.3443351, -1.8148437, 1.8120766
2: -0.6075819, -0.3018690, -0.6075076, -0.3019015, -0.2329538, 0.2338279
3: -0.3642939, 0.1840232, -0.3638524, 0.1830425, -0.4839600, 0.4849820
4: -2.4627304, -1.7337519, -2.4626563, -1.7337086, -0.2629327, 0.2643980
5: -0.5955282, 0.1139030, -0.5950338, 0.1130057, -0.6242146, 0.6253345
6: -3.3363454, -2.4709792, -3.3358142, -2.4723642, -0.2568888, 0.2594144
7: -0.3162112, 0.4358360, -0.3163916, 0.4367700, -0.6592060, 0.6575272
8: -4.6516843, -2.6777830, -4.6514173, -2.6783872, -1.0892048, 1.0810041
9: -1.8728881, -0.0033603, -1.8722229, -0.0040889, -1.4143977, 1.4125686

Time for backsubstitution: 6.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2344
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2343
type: A, layer: 1, pos: 2813
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3107
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 3035
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 3455
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3105
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2458
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2871
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2848
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 2340
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 885
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2462
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2909
type: A, layer: 1, pos: 2921
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3039

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1026295, upper bound: 0.1026078
time: 19.45 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027612, upper bound: 0.1026064
time: 25.84 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.9077964, -2.3777146, -3.9078031, -2.3774416, -1.3377337, 1.3377628
1: -1.6670403, 0.3554926, -1.6670446, 0.3557816, -1.8254557, 1.8250713
2: -0.6080965, -0.3018690, -0.6081280, -0.3018690, -0.2333415, 0.2340388
3: -0.3673836, 0.1840375, -0.3675501, 0.1840378, -0.4881738, 0.4881282
4: -2.4630766, -1.7337456, -2.4630685, -1.7337450, -0.2638393, 0.2641509
5: -0.5990319, 0.1139248, -0.5992237, 0.1139245, -0.6290820, 0.6290126
6: -3.3387434, -2.4709785, -3.3387849, -2.4709771, -0.2605492, 0.2583308
7: -0.3161464, 0.4358546, -0.3162621, 0.4358584, -0.6586441, 0.6578364
8: -4.6517587, -2.6734943, -4.6517625, -2.6733427, -1.0904603, 1.0895720
9: -1.8730035, 0.0012717, -1.8730087, 0.0014062, -1.4205823, 1.4193020

Time for backsubstitution: 6.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2344
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2343
type: A, layer: 1, pos: 2813
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3035
type: A, layer: 1, pos: 3107
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 3455
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3105
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2458
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2871
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2848
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 2340
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 885
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2462
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2909
type: A, layer: 1, pos: 2921
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3039

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1026309, upper bound: 0.1027612
time: 463.87 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027617, upper bound: 0.1027626
time: 261.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 732.36 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 732.36
Output dim: 3, lower bound: -0.1025550, upper bound: 0.1026053
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 732.36
Output dim: 3, lower bound: -0.1026861, upper bound: 0.1026045
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 732.36
Output dim: 3, lower bound: -0.1025566, upper bound: 0.1027615
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 732.36
Output dim: 3, lower bound: -0.1026886, upper bound: 0.1027594
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 732.36
Output dim: 3, lower bound: -0.1026295, upper bound: 0.1026078
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 732.36
Output dim: 3, lower bound: -0.1027612, upper bound: 0.1026064
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 732.36
Output dim: 3, lower bound: -0.1026309, upper bound: 0.1027612
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 732.36
Output dim: 3, lower bound: -0.1027617, upper bound: 0.1027626

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.9029007, -2.3997540, -3.9051371, -2.3985407, -1.3085823, 1.3120837
1: -1.6587076, 0.3262768, -1.6642442, 0.3277121, -1.7898598, 1.7938204
2: -0.6058502, -0.3022572, -0.6061006, -0.3019015, -0.2325724, 0.2321909
3: -0.3587059, 0.1817065, -0.3592035, 0.1830170, -0.4788787, 0.4783258
4: -2.4605303, -1.7345638, -2.4608736, -1.7337176, -0.2618721, 0.2610958
5: -0.5888183, 0.1111028, -0.5895650, 0.1129629, -0.6183448, 0.6177413
6: -3.3271782, -2.4745929, -3.3282628, -2.4723675, -0.2488994, 0.2471611
7: -0.3163125, 0.4370704, -0.3164941, 0.4367412, -0.6578979, 0.6583254
8: -4.6499743, -2.6852417, -4.6512728, -2.6845465, -1.0729311, 1.0758390
9: -1.8689179, -0.0119624, -1.8720551, -0.0114284, -1.4023275, 1.4047608

Time for backsubstitution: 6.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2343
type: B, layer: 1, pos: 2813
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3107
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 3455
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2458
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2871
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 885
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
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3034

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025045, upper bound: 0.1025182
time: 146.34 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1025121, upper bound: 0.1024985
time: 19.87 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.9056096, -2.3901348, -3.9053774, -2.3906446, -1.3218517, 1.3205261
1: -1.6631374, 0.3383784, -1.6643200, 0.3377771, -1.8042917, 1.8044710
2: -0.6070711, -0.3019880, -0.6070822, -0.3019015, -0.2332018, 0.2327293
3: -0.3625506, 0.1830977, -0.3623831, 0.1830308, -0.4819281, 0.4829351
4: -2.4619691, -1.7341036, -2.4620271, -1.7337120, -0.2613167, 0.2630962
5: -0.5935406, 0.1129436, -0.5933792, 0.1129874, -0.6219254, 0.6234045
6: -3.3343556, -2.4722073, -3.3341463, -2.4723666, -0.2458726, 0.2576646
7: -0.3168990, 0.4370819, -0.3169614, 0.4367592, -0.6578801, 0.6590469
8: -4.6506977, -2.6807556, -4.6513653, -2.6810040, -1.0804880, 1.0775659
9: -1.8706756, -0.0075774, -1.8721509, -0.0077853, -1.4079714, 1.4093370

Time for backsubstitution: 6.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2343
type: B, layer: 1, pos: 2813
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3107
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 3455
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2458
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2871
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 885
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
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3034

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1026360, upper bound: 0.1025192
time: 369.67 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1026349, upper bound: 0.1024968
time: 16.37 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.9031138, -2.3918824, -3.9074340, -2.3892465, -1.3155646, 1.3229318
1: -1.6587906, 0.3359013, -1.6669173, 0.3391304, -1.8004589, 1.8068013
2: -0.6063619, -0.3022572, -0.6067181, -0.3018690, -0.2329548, 0.2323938
3: -0.3617936, 0.1817215, -0.3628979, 0.1840123, -0.4830905, 0.4814707
4: -2.4608762, -1.7345575, -2.4612870, -1.7337557, -0.2627756, 0.2608459
5: -0.5923144, 0.1111248, -0.5937417, 0.1138827, -0.6232085, 0.6214199
6: -3.3295796, -2.4745910, -3.3312342, -2.4709804, -0.2525629, 0.2460845
7: -0.3162445, 0.4370882, -0.3163624, 0.4358289, -0.6573409, 0.6586368
8: -4.6500583, -2.6809535, -4.6516256, -2.6795111, -1.0741758, 1.0843921
9: -1.8690362, -0.0073309, -1.8728399, -0.0059414, -1.4084949, 1.4114780

Time for backsubstitution: 6.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2343
type: B, layer: 1, pos: 2813
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3107
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 3455
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2458
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2871
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 885
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
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3034

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025055, upper bound: 0.1026060
time: 85.29 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025081, upper bound: 0.1027113
time: 110.95 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.9058189, -2.3822618, -3.9076724, -2.3813503, -1.3288341, 1.3313751
1: -1.6632185, 0.3480039, -1.6669922, 0.3491969, -1.8148894, 1.8174496
2: -0.6075846, -0.3019880, -0.6077017, -0.3018690, -0.2335880, 0.2329383
3: -0.3656389, 0.1831125, -0.3660787, 0.1840269, -0.4861397, 0.4860815
4: -2.4623146, -1.7340978, -2.4624391, -1.7337490, -0.2622204, 0.2628464
5: -0.5970386, 0.1129655, -0.5975589, 0.1139074, -0.6267902, 0.6270839
6: -3.3367567, -2.4722061, -3.3371177, -2.4709799, -0.2495364, 0.2565882
7: -0.3168308, 0.4371002, -0.3168288, 0.4358473, -0.6573224, 0.6593583
8: -4.6507769, -2.6764665, -4.6517138, -2.6759644, -1.0817277, 1.0861177
9: -1.8707910, -0.0029440, -1.8729367, -0.0022979, -1.4141383, 1.4160552

Time for backsubstitution: 6.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2343
type: B, layer: 1, pos: 2813
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3107
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 3455
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2458
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2871
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 885
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
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3034

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1026387, upper bound: 0.1026072
time: 36.25 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1026394, upper bound: 0.1027116
time: 62.64 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.9048762, -2.3960595, -3.9052515, -2.3953102, -1.3167958, 1.3149304
1: -1.6625261, 0.3328528, -1.6642923, 0.3335366, -1.7997298, 1.7991419
2: -0.6062413, -0.3021383, -0.6064315, -0.3019015, -0.2320049, 0.2329417
3: -0.3601486, 0.1826302, -0.3604380, 0.1830265, -0.4797857, 0.4801349
4: -2.4611387, -1.7342123, -2.4613829, -1.7337132, -0.2613392, 0.2623531
5: -0.5904418, 0.1120594, -0.5909339, 0.1129783, -0.6191101, 0.6193819
6: -3.3287976, -2.4733651, -3.3296139, -2.4723654, -0.2483723, 0.2488006
7: -0.3154498, 0.4358217, -0.3157876, 0.4367501, -0.6583664, 0.6567054
8: -4.6509638, -2.6830549, -4.6513171, -2.6826119, -1.0813603, 1.0744152
9: -1.8711281, -0.0080504, -1.8721256, -0.0079803, -1.4086185, 1.4078131

Time for backsubstitution: 6.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2343
type: B, layer: 1, pos: 2813
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3107
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 3455
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2458
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2871
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 885
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
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3034

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025787, upper bound: 0.1025200
time: 174.93 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025792, upper bound: 0.1024973
time: 237.87 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.9075708, -2.3864400, -3.9054914, -2.3874149, -1.3300638, 1.3233695
1: -1.6669569, 0.3449545, -1.6643686, 0.3436031, -1.8141632, 1.8097906
2: -0.6074629, -0.3018690, -0.6074136, -0.3019015, -0.2326344, 0.2334884
3: -0.3639939, 0.1840213, -0.3636185, 0.1830409, -0.4828348, 0.4847440
4: -2.4625773, -1.7337533, -2.4625359, -1.7337074, -0.2607842, 0.2643533
5: -0.5951662, 0.1139002, -0.5947506, 0.1130029, -0.6226907, 0.6250455
6: -3.3359742, -2.4709795, -3.3354983, -2.4723647, -0.2453455, 0.2593042
7: -0.3160363, 0.4358341, -0.3162539, 0.4367679, -0.6583474, 0.6574278
8: -4.6516757, -2.6785684, -4.6514096, -2.6790671, -1.0889199, 1.0761359
9: -1.8728838, -0.0036635, -1.8722205, -0.0043368, -1.4142613, 1.4123907

Time for backsubstitution: 6.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2343
type: B, layer: 1, pos: 2813
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3107
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 3455
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2458
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2871
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 885
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
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3034

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027095, upper bound: 0.1025233
time: 14.72 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027103, upper bound: 0.1025006
time: 133.49 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.9050808, -2.3882024, -3.9075499, -2.3860159, -1.3237824, 1.3257818
1: -1.6626077, 0.3425093, -1.6669664, 0.3449821, -1.8103399, 1.8121371
2: -0.6067541, -0.3021383, -0.6070501, -0.3018690, -0.2323868, 0.2331505
3: -0.3632364, 0.1826451, -0.3641344, 0.1840231, -0.4839985, 0.4832801
4: -2.4614842, -1.7342062, -2.4617946, -1.7337506, -0.2622460, 0.2621060
5: -0.5939425, 0.1120817, -0.5951210, 0.1138989, -0.6239762, 0.6230595
6: -3.3311970, -2.4733639, -3.3325853, -2.4709778, -0.2520325, 0.2477166
7: -0.3153852, 0.4358402, -0.3156570, 0.4358380, -0.6578050, 0.6570144
8: -4.6510415, -2.6787672, -4.6516685, -2.6775680, -1.0826163, 1.0829802
9: -1.8712440, -0.0034189, -1.8729095, -0.0024867, -1.4148016, 1.4145460

Time for backsubstitution: 6.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2343
type: B, layer: 1, pos: 2813
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3107
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 3455
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2458
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2871
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 885
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
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3034

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025821, upper bound: 0.1026091
time: 148.66 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025830, upper bound: 0.1027112
time: 167.31 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.9077725, -2.3785810, -3.9077854, -2.3781195, -1.3370504, 1.3342233
1: -1.6670356, 0.3546114, -1.6670399, 0.3550487, -1.8247743, 1.8227844
2: -0.6079776, -0.3018690, -0.6080340, -0.3018690, -0.2330182, 0.2337018
3: -0.3670834, 0.1840359, -0.3673161, 0.1840369, -0.4870479, 0.4878901
4: -2.4629235, -1.7337469, -2.4629483, -1.7337458, -0.2616908, 0.2641064
5: -0.5986695, 0.1139221, -0.5989402, 0.1139227, -0.6275576, 0.6287228
6: -3.3383737, -2.4709785, -3.3384678, -2.4709775, -0.2490061, 0.2582207
7: -0.3159717, 0.4358513, -0.3161245, 0.4358559, -0.6577864, 0.6577361
8: -4.6517491, -2.6742797, -4.6517572, -2.6740222, -1.0901756, 1.0847037
9: -1.8729987, 0.0009670, -1.8730054, 0.0011568, -1.4204445, 1.4191236

Time for backsubstitution: 6.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2344
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2343
type: B, layer: 1, pos: 2813
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 3107
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 3455
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2458
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 271
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2871
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2340
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 885
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
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3494

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3034

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027131, upper bound: 0.1026116
time: 15.96 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027145, upper bound: 0.1027154
time: 22.14 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 44.79 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1025045, upper bound: 0.1025182
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1025121, upper bound: 0.1024985
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1026360, upper bound: 0.1025192
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1026349, upper bound: 0.1024968
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1025055, upper bound: 0.1026060
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1025081, upper bound: 0.1027113
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1026387, upper bound: 0.1026072
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1026394, upper bound: 0.1027116
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1025787, upper bound: 0.1025200
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1025792, upper bound: 0.1024973
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1027095, upper bound: 0.1025233
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1027103, upper bound: 0.1025006
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1025821, upper bound: 0.1026091
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1025830, upper bound: 0.1027112
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1027131, upper bound: 0.1026116
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 44.79
Output dim: 3, lower bound: -0.1027145, upper bound: 0.1027154

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.9028716, -2.4006538, -3.9051127, -2.3991237, -1.3077130, 1.3107719
1: -1.6586952, 0.3248405, -1.6642365, 0.3266659, -1.7888517, 1.7923422
2: -0.6057429, -0.3022572, -0.6059901, -0.3019015, -0.2323334, 0.2317715
3: -0.3581520, 0.1817035, -0.3586287, 0.1830137, -0.4783130, 0.4777631
4: -2.4604998, -1.7345651, -2.4608505, -1.7337184, -0.2617147, 0.2609947
5: -0.5882044, 0.1110995, -0.5889801, 0.1129592, -0.6177326, 0.6172432
6: -3.3265314, -2.4745944, -3.3274248, -2.4723697, -0.2483589, 0.2466576
7: -0.3162879, 0.4370636, -0.3164665, 0.4367341, -0.6578740, 0.6582966
8: -4.6499605, -2.6857152, -4.6512609, -2.6847982, -1.0718029, 1.0744762
9: -1.8689003, -0.0127206, -1.8720436, -0.0118904, -1.4016805, 1.4038377

Time for backsubstitution: 6.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2344
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2343
type: A, layer: 1, pos: 2813
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 3107
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 3035
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 3455
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3105
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2458
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2871
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 2848
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 2340
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 885
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2462
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2909
type: A, layer: 1, pos: 2921
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2344

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1023135, upper bound: 0.1025034
time: 171.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1024991, upper bound: 0.1025061
time: 31.36 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.9055815, -2.3910344, -3.9053550, -2.3912282, -1.3209820, 1.3192148
1: -1.6631255, 0.3369422, -1.6643119, 0.3367319, -1.8032832, 1.8029947
2: -0.6069627, -0.3019880, -0.6069710, -0.3019015, -0.2329619, 0.2323094
3: -0.3619954, 0.1830949, -0.3618078, 0.1830278, -0.4813612, 0.4823731
4: -2.4619396, -1.7341050, -2.4620037, -1.7337136, -0.2611596, 0.2629951
5: -0.5929251, 0.1129404, -0.5927946, 0.1129833, -0.6213131, 0.6229072
6: -3.3337085, -2.4722097, -3.3333077, -2.4723697, -0.2453322, 0.2571613
7: -0.3168749, 0.4370763, -0.3169332, 0.4367526, -0.6578560, 0.6590185
8: -4.6506853, -2.6812291, -4.6513553, -2.6812530, -1.0793595, 1.0762017
9: -1.8706584, -0.0083346, -1.8721390, -0.0082474, -1.4073243, 1.4084158

Time for backsubstitution: 6.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2344
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2343
type: A, layer: 1, pos: 2813
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 3107
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 3035
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 3455
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3105
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2458
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 271
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2871
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 2848
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 2340
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 885
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
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2462
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2909
type: A, layer: 1, pos: 2921
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3494

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2344

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1024458, upper bound: 0.1025054
time: 416.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1026290, upper bound: 0.1025065
time: 388.54 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 44.39 + 4089.92 = 4134.31 seconds
