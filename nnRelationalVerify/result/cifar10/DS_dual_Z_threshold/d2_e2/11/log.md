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
execution time: IAR + RelationalAnalysis = 7.39 + 36.56 = 43.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.1029240, upper bound: 0.1029258

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 885
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
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 3494

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 261

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029230, upper bound: 0.1028200
time: 14.75 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1028173, upper bound: 0.1029209
time: 178.08 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 192.91 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 192.91
Output dim: 3, lower bound: -0.1029230, upper bound: 0.1028200
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 192.91
Output dim: 3, lower bound: -0.1028173, upper bound: 0.1029209

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3411222, 1.3412113
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8301725, 1.8301649
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2350335, 0.2350438
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4898189, 0.4898101
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2636944, 0.2637695
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6311966, 0.6311884
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2613003, 0.2613534
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6589118, 0.6589174
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0977054, 1.0976982
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4218454, 1.4217949

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 885
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
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 3494

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2366

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027671, upper bound: 0.1028071
time: 16.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029084, upper bound: 0.1026609
time: 232.12 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3412113, 1.3411222
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8301644, 1.8301725
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2350438, 0.2350336
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4898101, 0.4898189
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2637694, 0.2636944
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6311884, 0.6311965
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2613534, 0.2613003
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6589174, 0.6589117
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0976982, 1.0977054
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4217944, 1.4218464

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 885
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
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 3494

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2366

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1026614, upper bound: 0.1028050
time: 145.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1028034, upper bound: 0.1027667
time: 218.65 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 370.55 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 370.55
Output dim: 3, lower bound: -0.1027671, upper bound: 0.1028071
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 370.55
Output dim: 3, lower bound: -0.1029084, upper bound: 0.1026609
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 370.55
Output dim: 3, lower bound: -0.1026614, upper bound: 0.1028050
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 370.55
Output dim: 3, lower bound: -0.1028034, upper bound: 0.1027667

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3396640, 1.3397331
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8295307, 1.8294935
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2341501, 0.2341998
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4894868, 0.4894940
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2630242, 0.2631255
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6305674, 0.6305819
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2574468, 0.2574886
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6588321, 0.6588593
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0948834, 1.0948853
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4211884, 1.4211287

Time for backsubstitution: 5.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 885
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
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 3494

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2365

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025761, upper bound: 0.1027965
time: 264.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027394, upper bound: 0.1024731
time: 161.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3396440, 1.3397532
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8295021, 1.8295221
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2341895, 0.2341604
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4895027, 0.4894781
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2630505, 0.2630993
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6305901, 0.6305592
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2574354, 0.2575000
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6588535, 0.6588378
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0948925, 1.0948758
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4211798, 1.4211373

Time for backsubstitution: 5.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 885
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
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 3494

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2365

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1026061, upper bound: 0.1026351
time: 39.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029017, upper bound: 0.1024725
time: 245.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3397532, 1.3396440
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8295231, 1.8295031
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2341604, 0.2341895
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4894781, 0.4895027
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2630993, 0.2630505
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6305592, 0.6305902
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2575000, 0.2574354
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6588377, 0.6588535
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0948763, 1.0948920
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4211369, 1.4211802

Time for backsubstitution: 5.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 885
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
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 3494

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2365

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1024727, upper bound: 0.1025030
time: 141.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1026350, upper bound: 0.1026082
time: 278.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3397331, 1.3396640
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8294945, 1.8295317
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2341998, 0.2341501
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4894940, 0.4894868
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2631256, 0.2630242
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6305820, 0.6305673
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2574885, 0.2574469
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6588593, 0.6588321
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0948853, 1.0948834
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4211283, 1.4211879

Time for backsubstitution: 5.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 885
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
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 3494

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2365

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1025023, upper bound: 0.1024721
time: 295.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027984, upper bound: 0.1025801
time: 18.10 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 319.24 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 319.24
Output dim: 3, lower bound: -0.1025761, upper bound: 0.1027965
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 319.24
Output dim: 3, lower bound: -0.1027394, upper bound: 0.1024731
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 319.24
Output dim: 3, lower bound: -0.1026061, upper bound: 0.1026351
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 319.24
Output dim: 3, lower bound: -0.1029017, upper bound: 0.1024725
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 319.24
Output dim: 3, lower bound: -0.1024727, upper bound: 0.1025030
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 319.24
Output dim: 3, lower bound: -0.1026350, upper bound: 0.1026082
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 319.24
Output dim: 3, lower bound: -0.1025023, upper bound: 0.1024721
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 319.24
Output dim: 3, lower bound: -0.1027984, upper bound: 0.1025801

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3378134, 1.3378873
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8287244, 1.8286386
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2333170, 0.2334317
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4890800, 0.4891129
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2624924, 0.2626430
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6297919, 0.6298546
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2529246, 0.2531109
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6586516, 0.6587257
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0914476, 1.0913038
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4203591, 1.4202561

Time for backsubstitution: 5.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 885
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
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 3494

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2367

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025336, upper bound: 0.1027768
time: 193.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025528, upper bound: 0.1027290
time: 43.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3378181, 1.3378816
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8286757, 1.8286881
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2333788, 0.2333667
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4891046, 0.4890871
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2625417, 0.2625938
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6298364, 0.6298064
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2530693, 0.2529663
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6586972, 0.6586785
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0913017, 1.0913911
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4203157, 1.4202952

Time for backsubstitution: 5.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 885
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
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 3494

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2367

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027014, upper bound: 0.1024853
time: 127.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027001, upper bound: 0.1024343
time: 77.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3377929, 1.3379073
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8286967, 1.8286672
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2333564, 0.2333891
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4890957, 0.4890959
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2625188, 0.2626167
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6298146, 0.6298283
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2529132, 0.2531224
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6586730, 0.6587029
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0913980, 1.0912948
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4203458, 1.4202647

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 885
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
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 3494

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2367

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025389, upper bound: 0.1025969
time: 127.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025887, upper bound: 0.1025969
time: 154.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3377981, 1.3379021
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8286471, 1.8287168
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2334213, 0.2333273
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4891216, 0.4890712
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2625679, 0.2625675
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6298627, 0.6297837
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2530578, 0.2529778
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6587200, 0.6586571
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0913107, 1.0914407
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4203076, 1.4203076

Time for backsubstitution: 6.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 885
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
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 3494

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2367

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1028325, upper bound: 0.1024507
time: 18.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1028833, upper bound: 0.1024279
time: 59.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3379073, 1.3377924
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8286672, 1.8286967
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2333891, 0.2333564
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4890959, 0.4890958
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2626167, 0.2625188
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6298282, 0.6298145
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2531224, 0.2529132
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6587030, 0.6586733
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0912945, 1.0913982
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4202647, 1.4203458

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 885
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
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 3494

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2367

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025970, upper bound: 0.1025903
time: 18.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025958, upper bound: 0.1025382
time: 26.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3378873, 1.3378134
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8286386, 1.8287253
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2334316, 0.2333170
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4891129, 0.4890800
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2626430, 0.2624925
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6298546, 0.6297919
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2531109, 0.2529246
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6587256, 0.6586514
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0913036, 1.0914478
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4202561, 1.4203591

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 885
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
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 3494

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2367

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027278, upper bound: 0.1025519
time: 85.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027779, upper bound: 0.1025325
time: 175.71 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 267.09 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 267.09
Output dim: 3, lower bound: -0.1025336, upper bound: 0.1027768
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 267.09
Output dim: 3, lower bound: -0.1025528, upper bound: 0.1027290
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 267.09
Output dim: 3, lower bound: -0.1027014, upper bound: 0.1024853
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 267.09
Output dim: 3, lower bound: -0.1027001, upper bound: 0.1024343
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 267.09
Output dim: 3, lower bound: -0.1025389, upper bound: 0.1025969
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 267.09
Output dim: 3, lower bound: -0.1025887, upper bound: 0.1025969
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 267.09
Output dim: 3, lower bound: -0.1028325, upper bound: 0.1024507
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 267.09
Output dim: 3, lower bound: -0.1028833, upper bound: 0.1024279
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 267.09
Output dim: 3, lower bound: -0.1025970, upper bound: 0.1025903
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 267.09
Output dim: 3, lower bound: -0.1025958, upper bound: 0.1025382
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 267.09
Output dim: 3, lower bound: -0.1027278, upper bound: 0.1025519
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 267.09
Output dim: 3, lower bound: -0.1027779, upper bound: 0.1025325

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3378029, 1.3378763
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8287234, 1.8286366
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2333177, 0.2334324
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4890800, 0.4891129
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2624890, 0.2626396
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6297920, 0.6298547
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2529276, 0.2531139
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6586506, 0.6587248
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0914199, 1.0912757
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4203587, 1.4202561

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 885
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
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 3494

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2126

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1024062, upper bound: 0.1024334
time: 256.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1025195, upper bound: 0.1025930
time: 295.35 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 557.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 557.70
Output dim: 3, lower bound: -0.1024062, upper bound: 0.1024334
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 557.70
Output dim: 3, lower bound: -0.1025195, upper bound: 0.1025930
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 557.70
Output dim: 3, lower bound: -0.1025528, upper bound: 0.1027290
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 557.70
Output dim: 3, lower bound: -0.1027014, upper bound: 0.1024853
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 557.70
Output dim: 3, lower bound: -0.1027001, upper bound: 0.1024343
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 557.70
Output dim: 3, lower bound: -0.1025389, upper bound: 0.1025969
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 557.70
Output dim: 3, lower bound: -0.1025887, upper bound: 0.1025969
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 557.70
Output dim: 3, lower bound: -0.1028325, upper bound: 0.1024507
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 557.70
Output dim: 3, lower bound: -0.1028833, upper bound: 0.1024279
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 557.70
Output dim: 3, lower bound: -0.1025970, upper bound: 0.1025903
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 557.70
Output dim: 3, lower bound: -0.1025958, upper bound: 0.1025382
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 557.70
Output dim: 3, lower bound: -0.1027278, upper bound: 0.1025519
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 557.70
Output dim: 3, lower bound: -0.1027779, upper bound: 0.1025325

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 43.95 + 3985.97 = 4029.92 seconds
