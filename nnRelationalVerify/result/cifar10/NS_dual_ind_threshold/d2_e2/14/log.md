## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 14)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.0399961638


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773981, 0.6773980)
1: (-4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633368, 0.9633367)
2: (-0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368)
3: (0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016447, 0.1016447)
4: (-1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309317, 0.2309317)
5: (0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145877, 0.1145877)
6: (-2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252266, 0.2252266)
7: (0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829536, 0.0829536)
8: (-4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319618, 0.6319618)
9: (-3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366515, 0.6366515)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.98 + 28.97 = 36.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0400272, upper bound: 0.0400339

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 424
type: A, layer: 1, pos: 428
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 420
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3432
type: A, layer: 1, pos: 3235
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 3433
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2898
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3520
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2957
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 2897
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3207
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 3002
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2311
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 2933
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 2268
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3117
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2258
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2935
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 439
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3434
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 424

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0396685, upper bound: 0.0400221
time: 34.62 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400236, upper bound: 0.0400260
time: 375.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 409.75 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 409.75
Output dim: 7, lower bound: -0.0396685, upper bound: 0.0400221
NS_A2, status: Status.UNKNOWN, split count: 1, time: 409.75
Output dim: 7, lower bound: -0.0400236, upper bound: 0.0400260

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.8291597, -2.7339005, -3.8294854, -2.7325444, -0.6757011, 0.6746018
1: -4.8637214, -3.0146656, -4.8640199, -3.0135977, -0.9618787, 0.9610884
2: -0.7749243, -0.5616050, -0.7749994, -0.5610541, -0.1349267, 0.1344012
3: 0.1319625, 0.4259911, 0.1295129, 0.4266344, -0.0964050, 0.0981836
4: -1.0536287, -0.4280396, -1.0549173, -0.4275921, -0.2280642, 0.2288848
5: 0.2932853, 0.5512214, 0.2909066, 0.5518124, -0.1096234, 0.1113686
6: -2.8281736, -1.9754183, -2.8310642, -1.9746590, -0.2191136, 0.2212856
7: 0.7683634, 1.2172642, 0.7676332, 1.2200117, -0.0791091, 0.0770282
8: -4.1436915, -2.5731347, -4.1439753, -2.5729480, -0.6293224, 0.6303617
9: -3.3512630, -1.9908001, -3.3514528, -1.9907675, -0.6353566, 0.6359431

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 428
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 420
type: B, layer: 1, pos: 424
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3432
type: B, layer: 1, pos: 3235
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2898
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3520
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2957
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 2897
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3207
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2311
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 2933
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 2268
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3117
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 439
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3434
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 428

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0396687, upper bound: 0.0397248
time: 101.55 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0396687, upper bound: 0.0400271
time: 104.20 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.8308635, -2.7324903, -3.8308637, -2.7324901, -0.6747597, 0.6773875
1: -4.8651872, -3.0135965, -4.8651867, -3.0135965, -0.9609804, 0.9633360
2: -0.7754345, -0.5610400, -0.7754350, -0.5610400, -0.1345061, 0.1354332
3: 0.1295117, 0.4294462, 0.1295117, 0.4294474, -0.1016248, 0.0962377
4: -1.0549197, -0.4258154, -1.0549197, -0.4258150, -0.2308989, 0.2283878
5: 0.2908299, 0.5544199, 0.2908298, 0.5544199, -0.1145686, 0.1094398
6: -2.8310714, -1.9714761, -2.8310716, -1.9714756, -0.2252179, 0.2188331
7: 0.7645768, 1.2200115, 0.7645719, 1.2200115, -0.0766876, 0.0829533
8: -4.1439815, -2.5724576, -4.1439805, -2.5724056, -0.6319588, 0.6291645
9: -3.3517330, -1.9907069, -3.3517334, -1.9906707, -0.6366291, 0.6358482

Time for backsubstitution: 5.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 428
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 420
type: B, layer: 1, pos: 424
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3432
type: B, layer: 1, pos: 3235
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2898
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3520
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2957
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 2897
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3207
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2311
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 2933
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 2268
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3117
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 439
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3434
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 428

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400233, upper bound: 0.0397276
time: 129.23 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400221, upper bound: 0.0400277
time: 45.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 180.34 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 180.34
Output dim: 7, lower bound: -0.0396687, upper bound: 0.0397248
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 180.34
Output dim: 7, lower bound: -0.0396687, upper bound: 0.0400271
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 180.34
Output dim: 7, lower bound: -0.0400233, upper bound: 0.0397276
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 180.34
Output dim: 7, lower bound: -0.0400221, upper bound: 0.0400277

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.8291595, -2.7339494, -3.8335516, -2.7325902, -0.6756897, 0.6788735
1: -4.8634620, -3.0146675, -4.8636961, -3.0109341, -0.9642462, 0.9600174
2: -0.7748934, -0.5616056, -0.7752907, -0.5585546, -0.1374081, 0.1339038
3: 0.1319626, 0.4259640, 0.1268346, 0.4268672, -0.0966281, 0.1001663
4: -1.0536275, -0.4281437, -1.0560839, -0.4270509, -0.2276206, 0.2310983
5: 0.2932855, 0.5512089, 0.2887796, 0.5519175, -0.1102877, 0.1129522
6: -2.8281741, -1.9754622, -2.8343797, -1.9741522, -0.2186457, 0.2246814
7: 0.7683764, 1.2172642, 0.7663471, 1.2248073, -0.0839113, 0.0774097
8: -4.1436911, -2.5731606, -4.1470933, -2.5729420, -0.6281568, 0.6335844
9: -3.3512464, -1.9908466, -3.3522871, -1.9907320, -0.6360251, 0.6361015

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 420
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 428
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3432
type: A, layer: 1, pos: 3235
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 3433
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2898
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3520
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2957
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 2897
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3207
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 3002
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2311
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 2933
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 2268
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3117
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 2258
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2935
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 439
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3434
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2417

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0395081, upper bound: 0.0400085
time: 147.82 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0396545, upper bound: 0.0400121
time: 147.91 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.8303237, -2.7339807, -3.8301711, -2.7343993, -0.6727570, 0.6757860
1: -4.8642077, -3.0136037, -4.8639350, -3.0136054, -0.9599967, 0.9620793
2: -0.7745936, -0.5610424, -0.7743582, -0.5610431, -0.1336320, 0.1343484
3: 0.1295190, 0.4283846, 0.1295212, 0.4280900, -0.0998979, 0.0946285
4: -1.0546569, -0.4262362, -1.0545825, -0.4263515, -0.2295513, 0.2272438
5: 0.2908748, 0.5538339, 0.2908873, 0.5536715, -0.1133900, 0.1082944
6: -2.8309581, -1.9726176, -2.8309262, -1.9729350, -0.2237164, 0.2176442
7: 0.7661853, 1.2200094, 0.7666317, 1.2200087, -0.0750008, 0.0808799
8: -4.1439743, -2.5734830, -4.1439734, -2.5737071, -0.6305045, 0.6280251
9: -3.3514197, -1.9908013, -3.3513336, -1.9907899, -0.6355954, 0.6346384

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 420
type: A, layer: 1, pos: 428
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3432
type: A, layer: 1, pos: 3235
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 3433
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2898
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3520
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2957
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 2897
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3207
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 3002
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2311
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 2933
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 2268
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3117
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2258
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2935
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 439
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3434
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2417

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0398613, upper bound: 0.0397164
time: 71.62 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400099, upper bound: 0.0397122
time: 159.00 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.8308632, -2.7325392, -3.8349295, -2.7325358, -0.6747480, 0.6816595
1: -4.8649273, -3.0135989, -4.8648643, -3.0109332, -0.9633474, 0.9622660
2: -0.7754035, -0.5610407, -0.7757264, -0.5585406, -0.1369874, 0.1349356
3: 0.1295118, 0.4294193, 0.1268333, 0.4296803, -0.1018509, 0.0982203
4: -1.0549188, -0.4259187, -1.0560862, -0.4252689, -0.2304634, 0.2306029
5: 0.2908301, 0.5544074, 0.2887026, 0.5545249, -0.1152322, 0.1110263
6: -2.8310714, -1.9715207, -2.8343861, -1.9709697, -0.2247500, 0.2222288
7: 0.7645903, 1.2200115, 0.7632860, 1.2248073, -0.0814899, 0.0833350
8: -4.1439815, -2.5724843, -4.1471000, -2.5723991, -0.6307934, 0.6323873
9: -3.3517175, -1.9907534, -3.3525639, -1.9906354, -0.6372988, 0.6360363

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 420
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 428
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 3432
type: A, layer: 1, pos: 3235
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 3433
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2898
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3520
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2957
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 2897
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3207
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 3002
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2311
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 2933
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2971
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 2268
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3117
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2258
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2935
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 439
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3434
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2417

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0398611, upper bound: 0.0400148
time: 49.98 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400084, upper bound: 0.0400125
time: 173.02 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 229.14 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 229.14
Output dim: 7, lower bound: -0.0395081, upper bound: 0.0400085
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 229.14
Output dim: 7, lower bound: -0.0396545, upper bound: 0.0400121
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 229.14
Output dim: 7, lower bound: -0.0398613, upper bound: 0.0397164
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 229.14
Output dim: 7, lower bound: -0.0400099, upper bound: 0.0397122
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 229.14
Output dim: 7, lower bound: -0.0398611, upper bound: 0.0400148
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 229.14
Output dim: 7, lower bound: -0.0400084, upper bound: 0.0400125

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.8274674, -2.7391806, -3.8335466, -2.7369947, -0.6684770, 0.6731584
1: -4.8588562, -3.0284128, -4.8636918, -3.0223489, -0.9462488, 0.9452505
2: -0.7748133, -0.5619151, -0.7752454, -0.5588053, -0.1368781, 0.1334723
3: 0.1319808, 0.4258641, 0.1268359, 0.4267882, -0.0965003, 0.1000523
4: -1.0523393, -0.4285765, -1.0550179, -0.4270846, -0.2262225, 0.2294859
5: 0.2935114, 0.5511434, 0.2889645, 0.5519153, -0.1100676, 0.1127047
6: -2.8276935, -1.9756581, -2.8339911, -1.9741542, -0.2181835, 0.2241392
7: 0.7704564, 1.2165954, 0.7680991, 1.2248026, -0.0816311, 0.0746203
8: -4.1393676, -2.5873313, -4.1470933, -2.5850761, -0.6088802, 0.6178141
9: -3.3462508, -2.0047669, -3.3518348, -2.0024962, -0.6167857, 0.6202638

Time for backsubstitution: 6.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 420
type: B, layer: 1, pos: 424
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3432
type: B, layer: 1, pos: 3235
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2898
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3520
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 2957
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2897
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 3207
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2311
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2933
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 2268
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 3117
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 439
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3434
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2642

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0394149, upper bound: 0.0399589
time: 63.04 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0394641, upper bound: 0.0399613
time: 15.34 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.8291588, -2.7341237, -3.8335514, -2.7327352, -0.6756336, 0.6709970
1: -4.8634610, -3.0151937, -4.8636966, -3.0113440, -0.9640708, 0.9400702
2: -0.7748895, -0.5616471, -0.7752877, -0.5585870, -0.1373692, 0.1333888
3: 0.1319627, 0.4259554, 0.1268348, 0.4268604, -0.0966243, 0.1000654
4: -1.0535748, -0.4281465, -1.0560427, -0.4270533, -0.2259102, 0.2310557
5: 0.2932922, 0.5512087, 0.2887848, 0.5519174, -0.1100424, 0.1129165
6: -2.8281703, -1.9754629, -2.8343763, -1.9741523, -0.2180719, 0.2246665
7: 0.7684596, 1.2172637, 0.7664118, 1.2248070, -0.0808437, 0.0773796
8: -4.1436911, -2.5737333, -4.1470942, -2.5733879, -0.6279684, 0.6122534
9: -3.3512073, -1.9913881, -3.3522558, -1.9911549, -0.6356446, 0.6157070

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 420
type: B, layer: 1, pos: 424
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3432
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3235
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2898
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3520
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 2957
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2897
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 3207
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2311
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2933
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 2268
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 3117
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 439
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3434
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2642

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0395660, upper bound: 0.0399616
time: 16.01 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0396012, upper bound: 0.0399601
time: 10.51 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.8303230, -2.7341545, -3.8301704, -2.7345450, -0.6727008, 0.6679105
1: -4.8642063, -3.0141301, -4.8639345, -3.0140154, -0.9598218, 0.9421315
2: -0.7745898, -0.5610840, -0.7743553, -0.5610754, -0.1335931, 0.1338337
3: 0.1295193, 0.4283759, 0.1295214, 0.4280830, -0.0998941, 0.0945277
4: -1.0546043, -0.4262390, -1.0545413, -0.4263539, -0.2278415, 0.2272012
5: 0.2908812, 0.5538338, 0.2908927, 0.5536715, -0.1131453, 0.1082587
6: -2.8309541, -1.9726186, -2.8309240, -1.9729357, -0.2231424, 0.2176293
7: 0.7662684, 1.2200092, 0.7666963, 1.2200085, -0.0719331, 0.0808498
8: -4.1439753, -2.5740566, -4.1439724, -2.5741541, -0.6303164, 0.6066940
9: -3.3513823, -1.9913437, -3.3513031, -1.9912124, -0.6352158, 0.6142440

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 420
type: B, layer: 1, pos: 424
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3432
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3235
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2898
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3520
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2957
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 2897
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3207
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2311
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 2933
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 2268
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3117
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 439
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3434
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2642

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399204, upper bound: 0.0396637
time: 16.26 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399554, upper bound: 0.0396669
time: 17.96 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.8291712, -2.7377696, -3.8349247, -2.7369406, -0.6675352, 0.6759446
1: -4.8603220, -3.0273454, -4.8648586, -3.0223475, -0.9453508, 0.9474986
2: -0.7753237, -0.5613502, -0.7756808, -0.5587913, -0.1364573, 0.1345044
3: 0.1295299, 0.4293191, 0.1268347, 0.4296010, -0.1017231, 0.0981061
4: -1.0536306, -0.4263521, -1.0550201, -0.4253026, -0.2290653, 0.2289903
5: 0.2910558, 0.5543419, 0.2888875, 0.5545229, -0.1150122, 0.1107789
6: -2.8305905, -1.9717152, -2.8339970, -1.9709718, -0.2242877, 0.2216865
7: 0.7666701, 1.2193429, 0.7650378, 1.2248026, -0.0792098, 0.0805456
8: -4.1396580, -2.5866537, -4.1470990, -2.5845332, -0.6115170, 0.6166169
9: -3.3467236, -2.0046740, -3.3521111, -2.0023999, -0.6180607, 0.6201953

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 420
type: B, layer: 1, pos: 424
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3432
type: B, layer: 1, pos: 3235
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2898
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3520
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 2957
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2897
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 3207
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2311
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2933
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 2268
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 3117
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 439
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3434
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2642

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0397690, upper bound: 0.0399577
time: 107.63 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0398170, upper bound: 0.0399618
time: 166.40 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.8308630, -2.7327127, -3.8349292, -2.7326808, -0.6746922, 0.6737827
1: -4.8649268, -3.0141249, -4.8648629, -3.0113440, -0.9631724, 0.9423184
2: -0.7753995, -0.5610821, -0.7757233, -0.5585729, -0.1369486, 0.1344209
3: 0.1295120, 0.4294106, 0.1268334, 0.4296735, -0.1018471, 0.0981194
4: -1.0548660, -0.4259214, -1.0560451, -0.4252707, -0.2287530, 0.2305602
5: 0.2908366, 0.5544071, 0.2887076, 0.5545249, -0.1149869, 0.1109907
6: -2.8310676, -1.9715204, -2.8343828, -1.9709697, -0.2241762, 0.2222139
7: 0.7646734, 1.2200112, 0.7633508, 1.2248070, -0.0784222, 0.0833049
8: -4.1439810, -2.5730574, -4.1470990, -2.5728462, -0.6306049, 0.6110560
9: -3.3516784, -1.9912953, -3.3525324, -1.9910586, -0.6369175, 0.6156416

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 420
type: B, layer: 1, pos: 424
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3432
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3235
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2898
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3520
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 2957
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2897
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 3207
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2311
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2933
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 2971
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 2268
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 3117
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 2935
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 439
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3434
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2642

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399211, upper bound: 0.0399623
time: 94.36 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399562, upper bound: 0.0399653
time: 41.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 141.46 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 141.46
Output dim: 7, lower bound: -0.0394149, upper bound: 0.0399589
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 141.46
Output dim: 7, lower bound: -0.0394641, upper bound: 0.0399613
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 141.46
Output dim: 7, lower bound: -0.0395660, upper bound: 0.0399616
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 141.46
Output dim: 7, lower bound: -0.0396012, upper bound: 0.0399601
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 141.46
Output dim: 7, lower bound: -0.0399204, upper bound: 0.0396637
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 141.46
Output dim: 7, lower bound: -0.0399554, upper bound: 0.0396669
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 141.46
Output dim: 7, lower bound: -0.0397690, upper bound: 0.0399577
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 141.46
Output dim: 7, lower bound: -0.0398170, upper bound: 0.0399618
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 141.46
Output dim: 7, lower bound: -0.0399211, upper bound: 0.0399623
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 141.46
Output dim: 7, lower bound: -0.0399562, upper bound: 0.0399653

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 36.95 + 2148.65 = 2185.60 seconds
