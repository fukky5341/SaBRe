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
execution time: IAR + RelationalAnalysis = 7.97 + 205.96 = 213.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0286829, upper bound: 0.0286831

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3351

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286810, upper bound: 0.0286820
time: 47.68 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286824, upper bound: 0.0286822
time: 3.94 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 51.70 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 51.70
Output dim: 6, lower bound: -0.0286810, upper bound: 0.0286820
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 51.70
Output dim: 6, lower bound: -0.0286824, upper bound: 0.0286822

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4873824, 0.4873824
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5610949, 0.5610948
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1172671, 0.1172671
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891932, 0.1891931
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2587516, 0.2587516
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1286354, 0.1286355
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1321569, 0.1321570
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5405293, 0.5405294
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4289690, 0.4289690
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3602481, 0.3602479

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3336

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286796, upper bound: 0.0286840
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286810, upper bound: 0.0286801
time: 157.48 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4873823, 0.4873824
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5610948, 0.5610951
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1172671, 0.1172671
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891931, 0.1891932
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2587516, 0.2587516
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1286355, 0.1286354
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1321570, 0.1321569
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5405294, 0.5405294
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4289690, 0.4289690
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3602479, 0.3602481

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3336

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286805, upper bound: 0.0286824
time: 4.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286823, upper bound: 0.0286808
time: 44.44 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 55.40 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 55.40
Output dim: 6, lower bound: -0.0286796, upper bound: 0.0286840
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 55.40
Output dim: 6, lower bound: -0.0286810, upper bound: 0.0286801
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 55.40
Output dim: 6, lower bound: -0.0286805, upper bound: 0.0286824
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 55.40
Output dim: 6, lower bound: -0.0286823, upper bound: 0.0286808

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4873836, 0.4873832
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5610583, 0.5610564
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1172707, 0.1172706
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891170, 0.1891148
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2587365, 0.2587370
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1286034, 0.1286033
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1321463, 0.1321463
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5405424, 0.5405435
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4289694, 0.4289691
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3601899, 0.3601871

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3200

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286763, upper bound: 0.0286819
time: 31.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286781, upper bound: 0.0286811
time: 3.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4873832, 0.4873835
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5610565, 0.5610582
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1172706, 0.1172707
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891148, 0.1891169
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2587370, 0.2587366
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1286032, 0.1286036
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1321462, 0.1321463
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5405434, 0.5405424
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4289691, 0.4289694
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3601873, 0.3601897

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3200

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286778, upper bound: 0.0286795
time: 48.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286795, upper bound: 0.0286790
time: 3.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4873835, 0.4873832
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5610582, 0.5610565
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1172707, 0.1172706
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891169, 0.1891148
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2587365, 0.2587370
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1286036, 0.1286032
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1321463, 0.1321462
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5405425, 0.5405434
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4289694, 0.4289691
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3601897, 0.3601872

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3200

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286773, upper bound: 0.0286794
time: 51.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286790, upper bound: 0.0286797
time: 3.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4873832, 0.4873836
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5610564, 0.5610583
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1172706, 0.1172707
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891148, 0.1891170
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2587370, 0.2587365
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1286033, 0.1286034
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1321463, 0.1321463
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5405435, 0.5405424
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4289691, 0.4289694
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3601872, 0.3601899

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3200

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286791, upper bound: 0.0286758
time: 330.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286808, upper bound: 0.0286767
time: 33.44 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 370.10 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 370.10
Output dim: 6, lower bound: -0.0286763, upper bound: 0.0286819
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 370.10
Output dim: 6, lower bound: -0.0286781, upper bound: 0.0286811
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 370.10
Output dim: 6, lower bound: -0.0286778, upper bound: 0.0286795
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 370.10
Output dim: 6, lower bound: -0.0286795, upper bound: 0.0286790
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 370.10
Output dim: 6, lower bound: -0.0286773, upper bound: 0.0286794
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 370.10
Output dim: 6, lower bound: -0.0286790, upper bound: 0.0286797
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 370.10
Output dim: 6, lower bound: -0.0286791, upper bound: 0.0286758
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 370.10
Output dim: 6, lower bound: -0.0286808, upper bound: 0.0286767

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4873832, 0.4873828
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5610529, 0.5610515
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1172677, 0.1172673
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891134, 0.1891112
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2587366, 0.2587370
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1286001, 0.1285996
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1321441, 0.1321442
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5405404, 0.5405415
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4289663, 0.4289664
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3601780, 0.3601768

Time for backsubstitution: 6.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3113

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286704, upper bound: 0.0286795
time: 276.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286756, upper bound: 0.0286744
time: 71.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4873831, 0.4873828
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5610534, 0.5610510
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1172675, 0.1172675
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891134, 0.1891112
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2587365, 0.2587370
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1285997, 0.1285999
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1321442, 0.1321441
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5405404, 0.5405415
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4289667, 0.4289660
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3601794, 0.3601753

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3113

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286721, upper bound: 0.0286798
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286773, upper bound: 0.0286750
time: 9.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4873828, 0.4873830
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5610511, 0.5610533
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1172675, 0.1172674
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891112, 0.1891133
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2587370, 0.2587365
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1285998, 0.1285998
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1321440, 0.1321443
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5405415, 0.5405405
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4289661, 0.4289666
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3601754, 0.3601793

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3113

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286719, upper bound: 0.0286778
time: 142.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286768, upper bound: 0.0286748
time: 4.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4873828, 0.4873831
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5610516, 0.5610528
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1172673, 0.1172677
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891112, 0.1891133
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2587369, 0.2587366
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1285994, 0.1286002
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1321441, 0.1321441
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5405415, 0.5405405
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4289665, 0.4289663
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3601769, 0.3601779

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3113

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286736, upper bound: 0.0286762
time: 37.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286786, upper bound: 0.0286710
time: 34.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4873831, 0.4873828
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5610528, 0.5610516
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1172677, 0.1172673
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891133, 0.1891112
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2587366, 0.2587369
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1286002, 0.1285994
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1321441, 0.1321441
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5405405, 0.5405415
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4289663, 0.4289664
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3601778, 0.3601769

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3113

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286715, upper bound: 0.0286800
time: 31.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286765, upper bound: 0.0286738
time: 116.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4873831, 0.4873828
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5610533, 0.5610511
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1172675, 0.1172675
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891133, 0.1891112
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2587365, 0.2587370
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1285998, 0.1285998
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1321442, 0.1321440
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5405405, 0.5405415
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4289667, 0.4289660
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3601793, 0.3601754

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3559

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3113

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286732, upper bound: 0.0286785
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286782, upper bound: 0.0286712
time: 50.33 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 60.59 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 60.59
Output dim: 6, lower bound: -0.0286704, upper bound: 0.0286795
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 60.59
Output dim: 6, lower bound: -0.0286756, upper bound: 0.0286744
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 60.59
Output dim: 6, lower bound: -0.0286721, upper bound: 0.0286798
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 60.59
Output dim: 6, lower bound: -0.0286773, upper bound: 0.0286750
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 60.59
Output dim: 6, lower bound: -0.0286719, upper bound: 0.0286778
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 60.59
Output dim: 6, lower bound: -0.0286768, upper bound: 0.0286748
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 60.59
Output dim: 6, lower bound: -0.0286736, upper bound: 0.0286762
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 60.59
Output dim: 6, lower bound: -0.0286786, upper bound: 0.0286710
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 60.59
Output dim: 6, lower bound: -0.0286715, upper bound: 0.0286800
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 60.59
Output dim: 6, lower bound: -0.0286765, upper bound: 0.0286738
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 60.59
Output dim: 6, lower bound: -0.0286732, upper bound: 0.0286785
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 60.59
Output dim: 6, lower bound: -0.0286782, upper bound: 0.0286712
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 60.59
Output dim: 6, lower bound: -0.0286791, upper bound: 0.0286758
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 60.59
Output dim: 6, lower bound: -0.0286808, upper bound: 0.0286767

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 213.92 + 1622.62 = 1836.54 seconds
