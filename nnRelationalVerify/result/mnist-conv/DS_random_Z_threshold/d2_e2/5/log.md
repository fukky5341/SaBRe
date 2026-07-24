## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.203377554


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2995014, 0.2995014)
1: (1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2688107, 0.2688106)
2: (-6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2405051, 0.2405052)
3: (-12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3479385, 0.3479385)
4: (-4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2939122, 0.2939122)
5: (-8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3066356, 0.3066357)
6: (-5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.3018336, 0.3018336)
7: (-6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2507591, 0.2507592)
8: (-1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2921203, 0.2921203)
9: (-7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3122545, 0.3122545)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.03 + 33.19 = 57.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.2163591, upper bound: 0.2163587

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6154
type: DSZ, layer: 1, pos: 5788
type: DSZ, layer: 1, pos: 5802
type: DSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6154

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163560, upper bound: 0.2157554
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2157558, upper bound: 0.2163556
time: 2.69 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.41 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.41
Output dim: 1, lower bound: -0.2163560, upper bound: 0.2157554
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.41
Output dim: 1, lower bound: -0.2157558, upper bound: 0.2163556

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.3032336, 0.2965822
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2694969, 0.2682710
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2394297, 0.2418805
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3524951, 0.3443821
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2945384, 0.2934198
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3051031, 0.3086046
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.3007305, 0.3032524
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2497332, 0.2520696
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2938668, 0.2907561
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3153673, 0.3098254

Time for backsubstitution: 22.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 5788
type: DSZ, layer: 1, pos: 5802

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2132833, upper bound: 0.2157471
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163471, upper bound: 0.2126833
time: 2.76 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2965822, 0.2995014
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2682709, 0.2688106
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2405051, 0.2394297
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3443822, 0.3479385
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2934200, 0.2939122
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3066356, 0.3051031
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.3018336, 0.3007304
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2507591, 0.2497333
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2907561, 0.2921203
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3098254, 0.3122545

Time for backsubstitution: 23.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5788
type: DSZ, layer: 1, pos: 5802
type: DSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5788

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2067402, upper bound: 0.2163476
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2157478, upper bound: 0.2073400
time: 2.73 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.56 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.56
Output dim: 1, lower bound: -0.2132833, upper bound: 0.2157471
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.56
Output dim: 1, lower bound: -0.2163471, upper bound: 0.2126833
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.56
Output dim: 1, lower bound: -0.2067402, upper bound: 0.2163476
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.56
Output dim: 1, lower bound: -0.2157478, upper bound: 0.2073400

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2913465, 0.2876527
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2578699, 0.2595468
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2396245, 0.2420196
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3464779, 0.3353206
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2806995, 0.2749671
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.2995024, 0.3043905
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2866054, 0.2844220
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2406406, 0.2452400
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2916917, 0.2872151
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3071046, 0.2988057

Time for backsubstitution: 22.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5788
type: DSZ, layer: 1, pos: 5802

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5788

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2042673, upper bound: 0.2157392
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2132750, upper bound: 0.2067318
time: 2.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2943041, 0.2846952
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2607728, 0.2566440
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2395687, 0.2420753
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3434336, 0.3383650
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2760859, 0.2795808
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3008890, 0.3030038
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2818999, 0.2891273
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2429036, 0.2429770
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2903258, 0.2885810
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3043476, 0.3015628

Time for backsubstitution: 22.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5788
type: DSZ, layer: 1, pos: 5802

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5788

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2073324, upper bound: 0.2126751
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163397, upper bound: 0.2036667
time: 2.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2879615, 0.2969806
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2551198, 0.2649603
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2367105, 0.2383209
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3441311, 0.3470970
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2896920, 0.2811549
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3058503, 0.3048753
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.3002619, 0.2953696
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2402891, 0.2466693
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2896219, 0.2882493
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3065385, 0.3010279

Time for backsubstitution: 22.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 5802

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2036671, upper bound: 0.2163393
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2067322, upper bound: 0.2132752
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2940614, 0.2908807
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2644207, 0.2556593
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2393964, 0.2356350
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3435403, 0.3476879
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2806627, 0.2901840
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3064080, 0.3043175
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2964729, 0.2991586
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2476952, 0.2392632
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2868847, 0.2909864
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2985986, 0.3089676

Time for backsubstitution: 23.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5802
type: DSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5802

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2142887, upper bound: 0.2073373
time: 2.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2157450, upper bound: 0.2058508
time: 2.78 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.66 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 1, lower bound: -0.2042673, upper bound: 0.2157392
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 1, lower bound: -0.2132750, upper bound: 0.2067318
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 1, lower bound: -0.2073324, upper bound: 0.2126751
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 1, lower bound: -0.2163397, upper bound: 0.2036667
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 1, lower bound: -0.2036671, upper bound: 0.2163393
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 1, lower bound: -0.2067322, upper bound: 0.2132752
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 1, lower bound: -0.2142887, upper bound: 0.2073373
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 1, lower bound: -0.2157450, upper bound: 0.2058508

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2827255, 0.2851318
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2447165, 0.2556933
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2358298, 0.2409108
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3462251, 0.3344791
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2769634, 0.2622058
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.2987157, 0.3041617
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2850304, 0.2790616
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2301708, 0.2421749
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2905580, 0.2833440
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3038149, 0.2875771

Time for backsubstitution: 22.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5802

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5802

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2027701, upper bound: 0.2157365
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2042646, upper bound: 0.2142803
time: 2.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2888253, 0.2790315
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2540175, 0.2463934
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2385153, 0.2382249
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3456364, 0.3350700
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2679380, 0.2712350
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.2992734, 0.3036039
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2812448, 0.2828505
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2375769, 0.2347702
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2878206, 0.2860812
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2958760, 0.2955168

Time for backsubstitution: 22.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5802

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5802

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2118141, upper bound: 0.2067292
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2132729, upper bound: 0.2052427
time: 2.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2856830, 0.2821738
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2476193, 0.2527915
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2357740, 0.2409661
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3431829, 0.3375235
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2723535, 0.2668195
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3001024, 0.3027749
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2803285, 0.2837667
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2324338, 0.2399133
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2891918, 0.2847099
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3010587, 0.2903341

Time for backsubstitution: 22.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5802

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5802

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2058429, upper bound: 0.2126724
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2073297, upper bound: 0.2112140
time: 2.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2917833, 0.2760741
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2569193, 0.2434905
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2384599, 0.2382807
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3425921, 0.3381122
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2633244, 0.2758448
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3006603, 0.3022171
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2765396, 0.2875522
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2398385, 0.2325072
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2864547, 0.2874473
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2931190, 0.2982730

Time for backsubstitution: 22.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5802

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5802

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2148803, upper bound: 0.2036640
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163365, upper bound: 0.2021697
time: 2.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2760741, 0.2880512
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2434906, 0.2562329
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2369054, 0.2384599
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3381121, 0.3380356
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2758447, 0.2626981
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3002484, 0.3006603
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2861335, 0.2765396
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2311967, 0.2398385
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2874473, 0.2847085
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2982730, 0.2900062

Time for backsubstitution: 22.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5802

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5802

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2021699, upper bound: 0.2163367
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2036644, upper bound: 0.2148805
time: 2.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2790315, 0.2850931
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2463933, 0.2533311
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2368496, 0.2385153
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3350700, 0.3410800
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2712351, 0.2673117
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3016350, 0.2992734
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2814317, 0.2812448
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2334597, 0.2375770
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2860812, 0.2860744
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2955168, 0.2927632

Time for backsubstitution: 22.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5802

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5802

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2052429, upper bound: 0.2132725
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2067295, upper bound: 0.2118141
time: 2.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2935197, 0.2890209
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2621679, 0.2550170
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2384024, 0.2353514
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3421975, 0.3429717
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2796429, 0.2866092
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3054159, 0.3040333
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2950597, 0.2942154
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2455552, 0.2386542
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2858662, 0.2906963
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2983372, 0.3080716

Time for backsubstitution: 22.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2112139, upper bound: 0.2073293
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2142807, upper bound: 0.2042642
time: 2.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2922013, 0.2903379
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2637787, 0.2534065
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2391128, 0.2346411
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3388243, 0.3463448
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2770880, 0.2891632
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3061237, 0.3033254
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2915292, 0.2977437
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2470858, 0.2371230
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2865947, 0.2899678
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2977029, 0.3087054

Time for backsubstitution: 22.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2126727, upper bound: 0.2058429
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2157369, upper bound: 0.2027700
time: 2.75 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.38 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2027701, upper bound: 0.2157365
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2042646, upper bound: 0.2142803
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2118141, upper bound: 0.2067292
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2132729, upper bound: 0.2052427
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2058429, upper bound: 0.2126724
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2073297, upper bound: 0.2112140
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2148803, upper bound: 0.2036640
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2163365, upper bound: 0.2021697
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2021699, upper bound: 0.2163367
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2036644, upper bound: 0.2148805
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2052429, upper bound: 0.2132725
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2067295, upper bound: 0.2118141
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2112139, upper bound: 0.2073293
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2142807, upper bound: 0.2042642
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2126727, upper bound: 0.2058429
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 1, lower bound: -0.2157369, upper bound: 0.2027700

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2821826, 0.2832718
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2424624, 0.2550500
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2348360, 0.2406273
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3448821, 0.3297627
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2759413, 0.2586311
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.2977235, 0.3038771
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2836136, 0.2741179
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2280296, 0.2415635
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2895395, 0.2830540
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3035519, 0.2866811

Time for backsubstitution: 23.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1514
type: DSZ, layer: 3, pos: 1117
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1803
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2873
type: DSZ, layer: 3, pos: 921

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1935

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2013246, upper bound: 0.2152689
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2023097, upper bound: 0.2152593
time: 2.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2808654, 0.2845911
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2440730, 0.2534393
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2355462, 0.2399169
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3415086, 0.3331358
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2733887, 0.2611859
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.2984314, 0.3031696
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2800866, 0.2776484
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2295610, 0.2400337
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2902678, 0.2823255
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3029189, 0.2873155

Time for backsubstitution: 22.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1117
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1514
type: DSZ, layer: 3, pos: 1803
type: DSZ, layer: 3, pos: 2873
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 907

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1117

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2036840, upper bound: 0.2138572
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2038415, upper bound: 0.2136999
time: 2.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2882835, 0.2771715
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2517634, 0.2457501
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2375215, 0.2379414
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3442932, 0.3303535
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2669166, 0.2676604
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.2982813, 0.3033193
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2798303, 0.2779070
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2354358, 0.2341594
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2868022, 0.2857912
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2956136, 0.2946208

Time for backsubstitution: 23.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 1514
type: DSZ, layer: 3, pos: 1803
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1117
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2873
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 2390

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2866

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2095269, upper bound: 0.2026312
time: 2.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2062597, upper bound: 0.2045556
time: 2.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2869653, 0.2784895
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2533742, 0.2441394
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2382318, 0.2372310
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3409200, 0.3337266
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2643634, 0.2702143
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.2989891, 0.3026118
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2763013, 0.2814355
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2369665, 0.2326290
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2875305, 0.2850627
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2949800, 0.2952546

Time for backsubstitution: 23.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1117
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1803
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1514
type: DSZ, layer: 3, pos: 2873

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1403

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2081340, upper bound: 0.1920192
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2006708, upper bound: 0.1998857
time: 2.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2851410, 0.2803138
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2453653, 0.2521482
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2347802, 0.2406826
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3418397, 0.3328070
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2713329, 0.2632449
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.2991103, 0.3024906
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2789136, 0.2788233
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2302927, 0.2393028
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2881733, 0.2844199
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3007965, 0.2894381

Time for backsubstitution: 23.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1803
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 1514
type: DSZ, layer: 3, pos: 2873
type: DSZ, layer: 3, pos: 1117
type: DSZ, layer: 3, pos: 2137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2390

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2005733, upper bound: 0.2093310
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2025018, upper bound: 0.2074027
time: 2.77 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 28.97 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.97
Output dim: 1, lower bound: -0.2013246, upper bound: 0.2152689
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.97
Output dim: 1, lower bound: -0.2023097, upper bound: 0.2152593
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.97
Output dim: 1, lower bound: -0.2036840, upper bound: 0.2138572
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.97
Output dim: 1, lower bound: -0.2038415, upper bound: 0.2136999
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.97
Output dim: 1, lower bound: -0.2095269, upper bound: 0.2026312
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.97
Output dim: 1, lower bound: -0.2062597, upper bound: 0.2045556
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.97
Output dim: 1, lower bound: -0.2081340, upper bound: 0.1920192
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.97
Output dim: 1, lower bound: -0.2006708, upper bound: 0.1998857
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.97
Output dim: 1, lower bound: -0.2005733, upper bound: 0.2093310
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.97
Output dim: 1, lower bound: -0.2025018, upper bound: 0.2074027
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.97
Output dim: 1, lower bound: -0.2073297, upper bound: 0.2112140
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.97
Output dim: 1, lower bound: -0.2148803, upper bound: 0.2036640
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.97
Output dim: 1, lower bound: -0.2163365, upper bound: 0.2021697
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.97
Output dim: 1, lower bound: -0.2021699, upper bound: 0.2163367
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.97
Output dim: 1, lower bound: -0.2036644, upper bound: 0.2148805
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.97
Output dim: 1, lower bound: -0.2052429, upper bound: 0.2132725
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.97
Output dim: 1, lower bound: -0.2067295, upper bound: 0.2118141
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.97
Output dim: 1, lower bound: -0.2112139, upper bound: 0.2073293
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.97
Output dim: 1, lower bound: -0.2142807, upper bound: 0.2042642
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.97
Output dim: 1, lower bound: -0.2126727, upper bound: 0.2058429
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.97
Output dim: 1, lower bound: -0.2157369, upper bound: 0.2027700

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.22 + 544.60 = 601.82 seconds
