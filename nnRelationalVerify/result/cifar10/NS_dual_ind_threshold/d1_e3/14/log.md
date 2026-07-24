## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 14)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0285688656


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4873917, 0.4873917)
1: (-4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5611193, 0.5611194)
2: (-0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1172676, 0.1172677)
3: (-1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1892098, 0.1892098)
4: (0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2587545, 0.2587545)
5: (-1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1286408, 0.1286408)
6: (0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1321599, 0.1321599)
7: (-0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5405351, 0.5405351)
8: (-5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4289716, 0.4289715)
9: (-4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3602765, 0.3602763)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.89 + 198.69 = 206.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0286829, upper bound: 0.0286831

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2784
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 2768
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 2733
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 2811
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2754
type: A, layer: 1, pos: 3218
type: A, layer: 1, pos: 2741
type: A, layer: 1, pos: 2755
type: A, layer: 1, pos: 2735
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 2752
type: A, layer: 1, pos: 2748
type: A, layer: 1, pos: 2734
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2751
type: A, layer: 1, pos: 2757
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 2758
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2779
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2747
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2785
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2740
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 3558
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2777
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 2875
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2773
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 2905
type: A, layer: 1, pos: 2860
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 3315
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2857
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3105
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2998
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 2806
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2790
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2803
type: A, layer: 1, pos: 2848
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 2902
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2818
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2750
type: A, layer: 1, pos: 2821
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2817
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 2594
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2759
type: A, layer: 1, pos: 2774
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2909
type: A, layer: 1, pos: 2915
type: A, layer: 1, pos: 2916
type: A, layer: 1, pos: 2917
type: A, layer: 1, pos: 2918
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2921
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 2924
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3314

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2363

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0285821, upper bound: 0.0286741
time: 20.78 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286728, upper bound: 0.0286722
time: 158.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 179.67 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 179.67
Output dim: 6, lower bound: -0.0285821, upper bound: 0.0286741
NS_A2, status: Status.UNKNOWN, split count: 1, time: 179.67
Output dim: 6, lower bound: -0.0286728, upper bound: 0.0286722

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.7386713, -2.7495337, -3.7386787, -2.7484934, -0.4789881, 0.4780675
1: -4.5826907, -3.2645454, -4.5826917, -3.2631788, -0.5495434, 0.5482045
2: -0.0731070, 0.1369032, -0.0731170, 0.1369400, -0.1166573, 0.1166685
3: -1.4028059, -1.0318210, -1.4030528, -1.0318162, -0.1870598, 0.1872368
4: 0.1337625, 0.4152864, 0.1336305, 0.4152865, -0.2575068, 0.2576364
5: -1.2806602, -0.9403848, -1.2808032, -0.9403833, -0.1262420, 0.1265201
6: 0.2618915, 0.4553722, 0.2616493, 0.4553723, -0.1300176, 0.1302578
7: -0.7888188, 0.0075190, -0.7888460, 0.0075197, -0.5401961, 0.5402328
8: -5.0408797, -4.2036490, -5.0408807, -4.2033458, -0.4259569, 0.4255671
9: -4.4275069, -3.6227503, -4.4275074, -3.6224058, -0.3577085, 0.3573651

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2784
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2768
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 2733
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 2811
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2754
type: B, layer: 1, pos: 3218
type: B, layer: 1, pos: 2741
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2755
type: B, layer: 1, pos: 2735
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 2752
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 2734
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 2757
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2758
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2779
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2785
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2777
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2905
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 3315
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2857
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2806
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 2902
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2818
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2750
type: B, layer: 1, pos: 2821
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2817
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 2594
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2759
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2915
type: B, layer: 1, pos: 2916
type: B, layer: 1, pos: 2917
type: B, layer: 1, pos: 2918
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 2924
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3314

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2349

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0285687, upper bound: 0.0286155
time: 8.40 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0285681, upper bound: 0.0286578
time: 64.65 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.7495065, -2.7404156, -3.7387397, -2.7403221, -0.4963886, 0.4789088
1: -4.5974064, -3.2509222, -4.5826950, -3.2508707, -0.5729092, 0.5494261
2: -0.0733339, 0.1371591, -0.0731672, 0.1371632, -0.1170654, 0.1168832
3: -1.4049661, -1.0292634, -1.4049793, -1.0317781, -0.1872617, 0.1907578
4: 0.1325540, 0.4153043, 0.1326116, 0.4152866, -0.2586707, 0.2586388
5: -1.2816507, -0.9390810, -1.2817336, -0.9403757, -0.1264491, 0.1299752
6: 0.2600269, 0.4553937, 0.2600980, 0.4553723, -0.1319166, 0.1319289
7: -0.7887833, 0.0074973, -0.7887877, 0.0075240, -0.5403439, 0.5403511
8: -5.0439529, -4.2011433, -5.0408821, -4.2011013, -0.4307804, 0.4258524
9: -4.4308209, -3.6198969, -4.4275088, -3.6198611, -0.3626162, 0.3576278

Time for backsubstitution: 6.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2784
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2768
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 2733
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 2811
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2754
type: B, layer: 1, pos: 3218
type: B, layer: 1, pos: 2741
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2755
type: B, layer: 1, pos: 2735
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 2752
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 2734
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 2757
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2758
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2779
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2785
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2777
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 2905
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 3315
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2857
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2806
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 2902
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2818
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2750
type: B, layer: 1, pos: 2821
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2817
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 2594
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2759
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2915
type: B, layer: 1, pos: 2916
type: B, layer: 1, pos: 2917
type: B, layer: 1, pos: 2918
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 2924
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3314

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2349

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286567, upper bound: 0.0286143
time: 33.59 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286566, upper bound: 0.0286563
time: 173.51 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 213.26 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 213.26
Output dim: 6, lower bound: -0.0285687, upper bound: 0.0286155
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 213.26
Output dim: 6, lower bound: -0.0285681, upper bound: 0.0286578
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 213.26
Output dim: 6, lower bound: -0.0286567, upper bound: 0.0286143
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 213.26
Output dim: 6, lower bound: -0.0286566, upper bound: 0.0286563

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3.7386014, -2.7543483, -3.7386031, -2.7535961, -0.4727628, 0.4724110
1: -4.5826874, -3.2706528, -4.5826893, -3.2699680, -0.5399783, 0.5396576
2: -0.0730422, 0.1367770, -0.0730442, 0.1367973, -0.1162242, 0.1162289
3: -1.4015971, -1.0318625, -1.4017038, -1.0318613, -0.1855087, 0.1855401
4: 0.1341597, 0.4152864, 0.1340744, 0.4152864, -0.2570924, 0.2571770
5: -1.2797091, -0.9403925, -1.2797356, -0.9403920, -0.1245874, 0.1246356
6: 0.2629822, 0.4553720, 0.2628816, 0.4553721, -0.1288992, 0.1290068
7: -0.7886064, 0.0075150, -0.7886055, 0.0075151, -0.5399617, 0.5399689
8: -5.0408783, -4.2052402, -5.0408783, -4.2051625, -0.4235486, 0.4234174
9: -4.4275055, -3.6248977, -4.4275064, -3.6248491, -0.3552262, 0.3551697

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2784
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 2768
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 2733
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 2811
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2754
type: A, layer: 1, pos: 3218
type: A, layer: 1, pos: 2741
type: A, layer: 1, pos: 2755
type: A, layer: 1, pos: 2735
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 2752
type: A, layer: 1, pos: 2748
type: A, layer: 1, pos: 2734
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2751
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2757
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 2758
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2779
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2747
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2785
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2740
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3558
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2777
type: A, layer: 1, pos: 2875
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2773
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 2905
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2860
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 3315
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2857
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3105
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2998
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 2806
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2790
type: A, layer: 1, pos: 2848
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2803
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2902
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2818
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2750
type: A, layer: 1, pos: 2821
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2817
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 2594
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2759
type: A, layer: 1, pos: 2774
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2909
type: A, layer: 1, pos: 2915
type: A, layer: 1, pos: 2916
type: A, layer: 1, pos: 2917
type: A, layer: 1, pos: 2918
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2921
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 2924
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3314

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 323

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0284760, upper bound: 0.0286063
time: 14.16 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0285650, upper bound: 0.0286060
time: 15.15 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.7386637, -2.7499080, -3.7465749, -2.7489109, -0.4731843, 0.4874686
1: -4.5826907, -3.2655606, -4.5929060, -3.2643323, -0.5404842, 0.5622946
2: -0.0730967, 0.1368535, -0.0732436, 0.1368858, -0.1164424, 0.1165994
3: -1.4027324, -1.0318244, -1.4029646, -1.0296857, -0.1895877, 0.1857381
4: 0.1338523, 0.4152865, 0.1336837, 0.4152870, -0.2574016, 0.2575774
5: -1.2799234, -0.9403856, -1.2799748, -0.9391836, -0.1288912, 0.1247096
6: 0.2623067, 0.4553721, 0.2620278, 0.4553834, -0.1297061, 0.1299419
7: -0.7887129, 0.0075169, -0.7887678, 0.0075182, -0.5401061, 0.5401626
8: -5.0408797, -4.2041974, -5.0431442, -4.2039337, -0.4235588, 0.4289706
9: -4.4275069, -3.6228561, -4.4310498, -3.6225269, -0.3553740, 0.3610204

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2784
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 2768
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 2733
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 2811
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2754
type: A, layer: 1, pos: 3218
type: A, layer: 1, pos: 2741
type: A, layer: 1, pos: 2755
type: A, layer: 1, pos: 2735
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 2752
type: A, layer: 1, pos: 2748
type: A, layer: 1, pos: 2734
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2751
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2757
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 2758
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2779
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2747
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2785
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2740
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3558
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2777
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 2875
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2773
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2905
type: A, layer: 1, pos: 2860
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 3315
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2857
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3105
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2998
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 2806
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2790
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2848
type: A, layer: 1, pos: 2803
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2902
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2818
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2750
type: A, layer: 1, pos: 2821
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2817
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 2594
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2759
type: A, layer: 1, pos: 2774
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2909
type: A, layer: 1, pos: 2915
type: A, layer: 1, pos: 2916
type: A, layer: 1, pos: 2917
type: A, layer: 1, pos: 2918
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2921
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 2924
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3314

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 323

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0284759, upper bound: 0.0286561
time: 8.15 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0285651, upper bound: 0.0286566
time: 3.14 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.7494440, -2.7444081, -3.7386694, -2.7448123, -0.4908227, 0.4733347
1: -4.5974045, -3.2568417, -4.5826907, -3.2576046, -0.5644386, 0.5408930
2: -0.0732715, 0.1370330, -0.0730970, 0.1370204, -0.1166401, 0.1164627
3: -1.4038986, -1.0292996, -1.4037592, -1.0318192, -0.1857419, 0.1892855
4: 0.1329232, 0.4153042, 0.1330276, 0.4152865, -0.2582841, 0.2582070
5: -1.2807018, -0.9390882, -1.2806691, -0.9403833, -0.1247964, 0.1282900
6: 0.2610615, 0.4553935, 0.2612671, 0.4553721, -0.1308529, 0.1307263
7: -0.7885755, 0.0074933, -0.7885528, 0.0075196, -0.5401138, 0.5400918
8: -5.0439520, -4.2026725, -5.0408797, -4.2028461, -0.4284930, 0.4237030
9: -4.4308205, -3.6218455, -4.4275079, -3.6220336, -0.3604233, 0.3554324

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2784
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 2768
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 2733
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 2811
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2754
type: A, layer: 1, pos: 3218
type: A, layer: 1, pos: 2741
type: A, layer: 1, pos: 2755
type: A, layer: 1, pos: 2735
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 2752
type: A, layer: 1, pos: 2748
type: A, layer: 1, pos: 2734
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2751
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2757
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 2758
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2779
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2747
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2785
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2740
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3558
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2777
type: A, layer: 1, pos: 2875
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2773
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 2905
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2860
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 3315
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2857
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3105
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2998
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 2806
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2790
type: A, layer: 1, pos: 2848
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2803
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2902
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2818
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2750
type: A, layer: 1, pos: 2821
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2817
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 2594
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2759
type: A, layer: 1, pos: 2774
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2909
type: A, layer: 1, pos: 2915
type: A, layer: 1, pos: 2916
type: A, layer: 1, pos: 2917
type: A, layer: 1, pos: 2918
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2921
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 2924
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3314

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 323

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0285618, upper bound: 0.0286064
time: 14.69 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286554, upper bound: 0.0286061
time: 40.19 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.7494950, -2.7412033, -3.7466309, -2.7412128, -0.4911253, 0.4882982
1: -4.5974064, -3.2532825, -4.5929089, -3.2532916, -0.5648584, 0.5634961
2: -0.0733209, 0.1371093, -0.0732912, 0.1371089, -0.1168479, 0.1168213
3: -1.4046905, -1.0292690, -1.4046900, -1.0296496, -0.1897864, 0.1894116
4: 0.1326824, 0.4153042, 0.1326976, 0.4152871, -0.2585348, 0.2585543
5: -1.2809101, -0.9390822, -1.2809012, -0.9391761, -0.1290966, 0.1283338
6: 0.2605138, 0.4553935, 0.2605509, 0.4553834, -0.1315315, 0.1315379
7: -0.7886714, 0.0074953, -0.7887038, 0.0075221, -0.5402482, 0.5402745
8: -5.0439520, -4.2020369, -5.0431461, -4.2020369, -0.4285057, 0.4292555
9: -4.4308214, -3.6201637, -4.4310508, -3.6201677, -0.3605694, 0.3612829

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2812
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2784
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 2768
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 2733
type: A, layer: 1, pos: 2756
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 2811
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2754
type: A, layer: 1, pos: 3218
type: A, layer: 1, pos: 2741
type: A, layer: 1, pos: 2755
type: A, layer: 1, pos: 2735
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 2752
type: A, layer: 1, pos: 2748
type: A, layer: 1, pos: 2734
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2794
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2751
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2757
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 2758
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2779
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2747
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2785
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2740
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3558
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2878
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2777
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 2875
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2908
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2773
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2905
type: A, layer: 1, pos: 2860
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 3315
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2857
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3105
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2998
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2958
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 2806
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2790
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2848
type: A, layer: 1, pos: 2803
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2902
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2818
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2750
type: A, layer: 1, pos: 2821
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2817
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 2594
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2759
type: A, layer: 1, pos: 2774
type: A, layer: 1, pos: 2789
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2909
type: A, layer: 1, pos: 2915
type: A, layer: 1, pos: 2916
type: A, layer: 1, pos: 2917
type: A, layer: 1, pos: 2918
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2921
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 2924
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3314

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 323

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0285617, upper bound: 0.0286557
time: 6.43 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286551, upper bound: 0.0286557
time: 25.20 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 37.75 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 37.75
Output dim: 6, lower bound: -0.0284760, upper bound: 0.0286063
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 37.75
Output dim: 6, lower bound: -0.0285650, upper bound: 0.0286060
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 37.75
Output dim: 6, lower bound: -0.0284759, upper bound: 0.0286561
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 37.75
Output dim: 6, lower bound: -0.0285651, upper bound: 0.0286566
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 37.75
Output dim: 6, lower bound: -0.0285618, upper bound: 0.0286064
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 37.75
Output dim: 6, lower bound: -0.0286554, upper bound: 0.0286061
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 37.75
Output dim: 6, lower bound: -0.0285617, upper bound: 0.0286557
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 37.75
Output dim: 6, lower bound: -0.0286551, upper bound: 0.0286557

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.7385023, -2.7548096, -3.7385187, -2.7539573, -0.4723450, 0.4719374
1: -4.5826321, -3.2713242, -4.5826421, -3.2705312, -0.5394609, 0.5390819
2: -0.0728603, 0.1362017, -0.0728896, 0.1363085, -0.1154480, 0.1153656
3: -1.4015682, -1.0320433, -1.4016795, -1.0320134, -0.1852041, 0.1852162
4: 0.1347230, 0.4152860, 0.1345494, 0.4152861, -0.2564922, 0.2566709
5: -1.2796860, -0.9416211, -1.2797160, -0.9414338, -0.1235857, 0.1235030
6: 0.2636009, 0.4552659, 0.2634018, 0.4552830, -0.1279195, 0.1280950
7: -0.7885308, 0.0041022, -0.7885410, 0.0046121, -0.5367758, 0.5362021
8: -5.0402513, -4.2052469, -5.0403385, -4.2051682, -0.4227394, 0.4227226
9: -4.4274116, -3.6248989, -4.4274268, -3.6248496, -0.3550768, 0.3550420

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2784
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2768
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 2733
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 2811
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2754
type: B, layer: 1, pos: 3218
type: B, layer: 1, pos: 2741
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2755
type: B, layer: 1, pos: 2735
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 2752
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 2734
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 2757
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2758
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2779
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2785
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2777
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 2905
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 3315
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2857
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2806
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2902
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2818
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2821
type: B, layer: 1, pos: 2750
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2817
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 2594
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2759
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2915
type: B, layer: 1, pos: 2916
type: B, layer: 1, pos: 2917
type: B, layer: 1, pos: 2918
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 2924
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3314

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2347

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0284671, upper bound: 0.0285175
time: 5.05 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0284667, upper bound: 0.0286017
time: 4.10 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.7398465, -2.7543771, -3.7385979, -2.7536497, -0.4740062, 0.4720524
1: -4.5846200, -3.2700229, -4.5826869, -3.2699795, -0.5413656, 0.5400094
2: -0.0759537, 0.1367520, -0.0730410, 0.1367708, -0.1198056, 0.1158273
3: -1.4020368, -1.0320995, -1.4017041, -1.0321047, -0.1855782, 0.1855384
4: 0.1337844, 0.4155254, 0.1341072, 0.4152864, -0.2573681, 0.2574548
5: -1.2836797, -0.9405710, -1.2797353, -0.9405722, -0.1280551, 0.1237194
6: 0.2632033, 0.4564918, 0.2631233, 0.4553720, -0.1282323, 0.1295566
7: -0.8002805, 0.0080001, -0.7886043, 0.0073779, -0.5520978, 0.5385576
8: -5.0406914, -4.2030311, -5.0403566, -4.2051640, -0.4231586, 0.4252100
9: -4.4272118, -3.6247902, -4.4272022, -3.6248488, -0.3551077, 0.3553396

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2784
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2768
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 2733
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 2811
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2754
type: B, layer: 1, pos: 3218
type: B, layer: 1, pos: 2741
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2755
type: B, layer: 1, pos: 2735
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 2752
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 2734
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2757
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2758
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2779
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2785
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2777
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 2905
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 3315
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2857
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2806
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2902
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2818
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2750
type: B, layer: 1, pos: 2821
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2817
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 2594
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2759
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2915
type: B, layer: 1, pos: 2916
type: B, layer: 1, pos: 2917
type: B, layer: 1, pos: 2918
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 2924
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3314

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2347

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0285589, upper bound: 0.0285162
time: 59.58 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0285588, upper bound: 0.0286002
time: 48.81 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.7385650, -2.7503703, -3.7464960, -2.7492983, -0.4727669, 0.4870027
1: -4.5826349, -3.2662206, -4.5928645, -3.2648859, -0.5399644, 0.5617094
2: -0.0729140, 0.1362557, -0.0730935, 0.1363716, -0.1156663, 0.1157250
3: -1.4027035, -1.0320051, -1.4029406, -1.0298347, -0.1893118, 0.1854146
4: 0.1344342, 0.4152862, 0.1341736, 0.4152868, -0.2567866, 0.2570531
5: -1.2799006, -0.9416143, -1.2799563, -0.9402173, -0.1280072, 0.1235773
6: 0.2629452, 0.4552659, 0.2625659, 0.4552945, -0.1286760, 0.1289773
7: -0.7886367, 0.0041039, -0.7887061, 0.0046183, -0.5369197, 0.5363985
8: -5.0402522, -4.2042036, -5.0426035, -4.2039385, -0.4227502, 0.4283356
9: -4.4274130, -3.6228571, -4.4309702, -3.6225274, -0.3552245, 0.3609087

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2784
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2768
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 2733
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 2811
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2754
type: B, layer: 1, pos: 3218
type: B, layer: 1, pos: 2741
type: B, layer: 1, pos: 2755
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2735
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 2752
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 2734
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 2757
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2758
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2779
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2785
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2777
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 2905
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 3315
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2857
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2806
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2902
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2818
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2821
type: B, layer: 1, pos: 2750
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2817
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 2594
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2759
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2915
type: B, layer: 1, pos: 2916
type: B, layer: 1, pos: 2917
type: B, layer: 1, pos: 2918
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 2924
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3314

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2347

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0284668, upper bound: 0.0285640
time: 26.88 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0284667, upper bound: 0.0286488
time: 58.51 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.7399063, -2.7499578, -3.7465644, -2.7489614, -0.4744244, 0.4871117
1: -4.5846214, -3.2649689, -4.5928988, -3.2643661, -0.5418726, 0.5626155
2: -0.0760008, 0.1368331, -0.0732397, 0.1368657, -0.1199796, 0.1161702
3: -1.4031663, -1.0320626, -1.4029645, -1.0299422, -0.1895120, 0.1857322
4: 0.1334813, 0.4155254, 0.1337118, 0.4152869, -0.2576804, 0.2578592
5: -1.2836778, -0.9405646, -1.2799746, -0.9396061, -0.1320013, 0.1237829
6: 0.2623848, 0.4564918, 0.2621475, 0.4553833, -0.1290642, 0.1305882
7: -0.8003325, 0.0080026, -0.7887661, 0.0073804, -0.5522016, 0.5387506
8: -5.0406919, -4.2021723, -5.0425129, -4.2039347, -0.4231694, 0.4303030
9: -4.4272122, -3.6227739, -4.4307132, -3.6225269, -0.3552542, 0.3610870

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2784
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2768
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 2733
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 2811
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2754
type: B, layer: 1, pos: 3218
type: B, layer: 1, pos: 2741
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2755
type: B, layer: 1, pos: 2735
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 2752
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 2734
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2757
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2758
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2779
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2785
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2777
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 2905
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 3315
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2857
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2806
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2902
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2818
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2821
type: B, layer: 1, pos: 2750
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2817
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 2594
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2759
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2915
type: B, layer: 1, pos: 2916
type: B, layer: 1, pos: 2917
type: B, layer: 1, pos: 2918
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 2924
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3314

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2347

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0285591, upper bound: 0.0285646
time: 3.44 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0285589, upper bound: 0.0286490
time: 203.60 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.7493443, -2.7448702, -3.7385850, -2.7451735, -0.4903908, 0.4728591
1: -4.5973501, -3.2575135, -4.5826454, -3.2581687, -0.5638655, 0.5403160
2: -0.0730897, 0.1364577, -0.0729421, 0.1365317, -0.1158616, 0.1155993
3: -1.4038687, -1.0294553, -1.4037342, -1.0319721, -0.1854368, 0.1889810
4: 0.1334881, 0.4153039, 0.1335036, 0.4152863, -0.2576814, 0.2576995
5: -1.2806790, -0.9403158, -1.2806493, -0.9414251, -0.1237946, 0.1271658
6: 0.2616935, 0.4552875, 0.2617981, 0.4552831, -0.1298642, 0.1298106
7: -0.7884986, 0.0040807, -0.7884872, 0.0046161, -0.5369278, 0.5363248
8: -5.0433731, -4.2026782, -5.0403404, -4.2028503, -0.4277384, 0.4230081
9: -4.4307380, -3.6218472, -4.4274273, -3.6220338, -0.3602892, 0.3553048

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2784
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2768
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 2733
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 2811
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2754
type: B, layer: 1, pos: 3218
type: B, layer: 1, pos: 2741
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2755
type: B, layer: 1, pos: 2735
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 2752
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 2734
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 2757
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2758
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2779
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2785
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2777
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2905
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 3315
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2857
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2806
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2902
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2818
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2750
type: B, layer: 1, pos: 2821
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2817
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 2594
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2759
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2915
type: B, layer: 1, pos: 2916
type: B, layer: 1, pos: 2917
type: B, layer: 1, pos: 2918
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 2924
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3314

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2347

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0285531, upper bound: 0.0285168
time: 92.28 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0285526, upper bound: 0.0286023
time: 5.81 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.7506893, -2.7444358, -3.7386637, -2.7448647, -0.4921592, 0.4729761
1: -4.5993381, -3.2562094, -4.5826883, -3.2576146, -0.5660992, 0.5412412
2: -0.0761791, 0.1370078, -0.0730939, 0.1369940, -0.1202202, 0.1160636
3: -1.4042345, -1.0296021, -1.4037590, -1.0320628, -0.1858056, 0.1893481
4: 0.1325433, 0.4155433, 0.1330582, 0.4152864, -0.2585645, 0.2584880
5: -1.2846649, -0.9392721, -1.2806690, -0.9405636, -0.1282620, 0.1273811
6: 0.2612494, 0.4565143, 0.2614881, 0.4553720, -0.1301871, 0.1312823
7: -0.8002368, 0.0079792, -0.7885519, 0.0073823, -0.5522431, 0.5386809
8: -5.0436411, -4.2006960, -5.0403581, -4.2028465, -0.4280912, 0.4254953
9: -4.4304957, -3.6217875, -4.4272041, -3.6220338, -0.3603099, 0.3556023

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2784
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2768
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 2733
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 2811
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2754
type: B, layer: 1, pos: 3218
type: B, layer: 1, pos: 2741
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2755
type: B, layer: 1, pos: 2735
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 2752
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 2734
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2757
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2758
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2779
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2785
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2777
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2905
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 3315
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2857
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2806
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 2902
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2818
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2750
type: B, layer: 1, pos: 2821
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2817
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 2594
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2759
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2915
type: B, layer: 1, pos: 2916
type: B, layer: 1, pos: 2917
type: B, layer: 1, pos: 2918
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 2924
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3314

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2347

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286488, upper bound: 0.0285165
time: 38.90 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286487, upper bound: 0.0286001
time: 90.71 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.7493954, -2.7416747, -3.7465520, -2.7416086, -0.4906928, 0.4878315
1: -4.5973516, -3.2539549, -4.5928669, -3.2538567, -0.5642825, 0.5629095
2: -0.0731382, 0.1365116, -0.0731407, 0.1365947, -0.1160689, 0.1159469
3: -1.4046609, -1.0294244, -1.4046652, -1.0297996, -0.1895105, 0.1891068
4: 0.1332704, 0.4153040, 0.1331919, 0.4152869, -0.2579124, 0.2580282
5: -1.2808872, -0.9403099, -1.2808824, -0.9402097, -0.1282125, 0.1272099
6: 0.2611654, 0.4552875, 0.2610990, 0.4552944, -0.1304860, 0.1305626
7: -0.7885946, 0.0040824, -0.7886411, 0.0046221, -0.5370620, 0.5365098
8: -5.0433750, -4.2020431, -5.0426044, -4.2020411, -0.4277516, 0.4286205
9: -4.4307389, -3.6201649, -4.4309716, -3.6201682, -0.3604351, 0.3611711

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2784
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2768
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 2733
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 2811
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2754
type: B, layer: 1, pos: 3218
type: B, layer: 1, pos: 2741
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2755
type: B, layer: 1, pos: 2735
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 2752
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 2734
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 2757
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2758
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2779
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2785
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2777
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2905
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 3315
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2857
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2806
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2902
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2818
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2750
type: B, layer: 1, pos: 2821
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2817
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 2594
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2759
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2915
type: B, layer: 1, pos: 2916
type: B, layer: 1, pos: 2917
type: B, layer: 1, pos: 2918
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 2924
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3314

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2347

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0285524, upper bound: 0.0285651
time: 88.02 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0285528, upper bound: 0.0286488
time: 156.12 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.7507381, -2.7412286, -3.7466207, -2.7412434, -0.4924604, 0.4879425
1: -4.5993409, -3.2526622, -4.5929031, -3.2533007, -0.5665219, 0.5638133
2: -0.0762210, 0.1370889, -0.0732876, 0.1370888, -0.1203817, 0.1163950
3: -1.4050183, -1.0295722, -1.4046898, -1.0299058, -0.1897091, 0.1894703
4: 0.1322982, 0.4155434, 0.1327170, 0.4152871, -0.2588200, 0.2588401
5: -1.2846565, -0.9392669, -1.2809005, -0.9395982, -0.1322051, 0.1274145
6: 0.2605594, 0.4565142, 0.2606426, 0.4553834, -0.1308837, 0.1322165
7: -0.8002789, 0.0079811, -0.7887017, 0.0073842, -0.5523365, 0.5388625
8: -5.0436416, -4.2002258, -5.0425143, -4.2020369, -0.4281042, 0.4305878
9: -4.4304962, -3.6201308, -4.4307146, -3.6201677, -0.3604547, 0.3613495

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2812
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2784
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 2768
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 2733
type: B, layer: 1, pos: 2756
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 2811
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 2754
type: B, layer: 1, pos: 3218
type: B, layer: 1, pos: 2741
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2755
type: B, layer: 1, pos: 2735
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 2752
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 2734
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2794
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2757
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2758
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2779
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2785
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3558
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2878
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2777
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2908
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2773
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2905
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 3315
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2857
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2998
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2958
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2806
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 2803
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2902
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2818
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2750
type: B, layer: 1, pos: 2821
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2817
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 2594
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2759
type: B, layer: 1, pos: 2774
type: B, layer: 1, pos: 2789
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2909
type: B, layer: 1, pos: 2915
type: B, layer: 1, pos: 2916
type: B, layer: 1, pos: 2917
type: B, layer: 1, pos: 2918
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 2924
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3314

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2347

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286487, upper bound: 0.0285660
time: 3.73 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286489, upper bound: 0.0286497
time: 82.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 92.69 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0284671, upper bound: 0.0285175
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0284667, upper bound: 0.0286017
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0285589, upper bound: 0.0285162
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0285588, upper bound: 0.0286002
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0284668, upper bound: 0.0285640
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0284667, upper bound: 0.0286488
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0285591, upper bound: 0.0285646
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0285589, upper bound: 0.0286490
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0285531, upper bound: 0.0285168
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0285526, upper bound: 0.0286023
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0286488, upper bound: 0.0285165
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0286487, upper bound: 0.0286001
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0285524, upper bound: 0.0285651
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0285528, upper bound: 0.0286488
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0286487, upper bound: 0.0285660
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 92.69
Output dim: 6, lower bound: -0.0286489, upper bound: 0.0286497

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 206.58 + 1640.72 = 1847.31 seconds
