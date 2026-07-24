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
execution time: IAR + RelationalAnalysis = 21.43 + 32.33 = 53.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.2163591, upper bound: 0.2163587

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 5802
type: DSZ, layer: 1, pos: 5788
type: DSZ, layer: 1, pos: 6154

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2132870, upper bound: 0.2163504
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163508, upper bound: 0.2132866
time: 2.69 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.53 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.53
Output dim: 1, lower bound: -0.2132870, upper bound: 0.2163504
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.53
Output dim: 1, lower bound: -0.2163508, upper bound: 0.2132866

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2876145, 0.2905720
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2571836, 0.2600864
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2407001, 0.2406443
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3419214, 0.3388771
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2800730, 0.2754594
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3010349, 0.3024216
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2877085, 0.2830033
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2416667, 0.2439296
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2899454, 0.2885795
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3039919, 0.3012350

Time for backsubstitution: 19.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5802
type: DSZ, layer: 1, pos: 5788
type: DSZ, layer: 1, pos: 6154

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 5802

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2118257, upper bound: 0.2163477
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2132839, upper bound: 0.2148914
time: 2.77 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2905720, 0.2876145
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2600864, 0.2571836
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2406443, 0.2407001
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3388771, 0.3419214
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2754594, 0.2800730
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3024217, 0.3010349
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2830033, 0.2877086
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2439296, 0.2416666
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2885795, 0.2899454
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3012350, 0.3039919

Time for backsubstitution: 20.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5802
type: DSZ, layer: 1, pos: 5788
type: DSZ, layer: 1, pos: 6154

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 5802

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2148918, upper bound: 0.2132839
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163477, upper bound: 0.2118254
time: 2.94 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 26.80 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 26.80
Output dim: 1, lower bound: -0.2118257, upper bound: 0.2163477
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 26.80
Output dim: 1, lower bound: -0.2132839, upper bound: 0.2148914
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 26.80
Output dim: 1, lower bound: -0.2148918, upper bound: 0.2132839
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 26.80
Output dim: 1, lower bound: -0.2163477, upper bound: 0.2118254

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2870715, 0.2887120
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2549304, 0.2594440
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2397064, 0.2403610
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3405784, 0.3341609
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2790520, 0.2718850
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3000427, 0.3021370
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2862946, 0.2780601
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2395263, 0.2433196
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2889268, 0.2882894
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3037298, 0.3003390

Time for backsubstitution: 20.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5788
type: DSZ, layer: 1, pos: 6154

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 5788

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2027731, upper bound: 0.2163397
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2118176, upper bound: 0.2073324
time: 2.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2857544, 0.2900300
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2565410, 0.2578333
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2404166, 0.2396505
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3372053, 0.3375342
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2764988, 0.2744398
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3007506, 0.3014295
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2827655, 0.2815906
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2410575, 0.2417892
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2896553, 0.2875611
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3030961, 0.3009734

Time for backsubstitution: 20.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5788
type: DSZ, layer: 1, pos: 6154

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 5788

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2042676, upper bound: 0.2148835
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2132758, upper bound: 0.2058460
time: 2.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2900300, 0.2857544
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2578333, 0.2565410
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2396506, 0.2404166
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3375341, 0.3372053
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2744398, 0.2764988
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3014295, 0.3007506
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2815906, 0.2827656
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2417892, 0.2410575
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2875609, 0.2896553
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3009734, 0.3030961

Time for backsubstitution: 20.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5788
type: DSZ, layer: 1, pos: 6154

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 5788

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2058459, upper bound: 0.2132756
time: 2.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2148834, upper bound: 0.2042671
time: 2.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2887119, 0.2870715
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2594439, 0.2549304
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2403610, 0.2397063
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3341609, 0.3405785
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2718852, 0.2790519
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3021370, 0.3000427
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2780603, 0.2862946
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2433196, 0.2395262
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2882894, 0.2889268
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3003390, 0.3037298

Time for backsubstitution: 21.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5788
type: DSZ, layer: 1, pos: 6154

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 5788

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2073325, upper bound: 0.2118172
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163399, upper bound: 0.2027730
time: 2.74 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 26.78 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.78
Output dim: 1, lower bound: -0.2027731, upper bound: 0.2163397
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.78
Output dim: 1, lower bound: -0.2118176, upper bound: 0.2073324
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.78
Output dim: 1, lower bound: -0.2042676, upper bound: 0.2148835
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.78
Output dim: 1, lower bound: -0.2132758, upper bound: 0.2058460
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.78
Output dim: 1, lower bound: -0.2058459, upper bound: 0.2132756
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.78
Output dim: 1, lower bound: -0.2148834, upper bound: 0.2042671
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.78
Output dim: 1, lower bound: -0.2073325, upper bound: 0.2118172
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.78
Output dim: 1, lower bound: -0.2163399, upper bound: 0.2027730

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2784504, 0.2861911
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2417762, 0.2555896
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2359115, 0.2392521
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3403256, 0.3333191
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2753146, 0.2591232
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.2992562, 0.3019083
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2847167, 0.2726992
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2290557, 0.2402532
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2877932, 0.2844185
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.3004392, 0.2891104

Time for backsubstitution: 20.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6154

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 6154

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2027701, upper bound: 0.2157365
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2021699, upper bound: 0.2163367
time: 2.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2845514, 0.2800908
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2510771, 0.2462897
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2385970, 0.2365662
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3397367, 0.3339100
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2662899, 0.2681524
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.2998141, 0.3013505
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2809335, 0.2764884
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2364617, 0.2328491
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2850559, 0.2871556
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2925010, 0.2970500

Time for backsubstitution: 20.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6154

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6154

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2118141, upper bound: 0.2067292
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2112139, upper bound: 0.2073293
time: 2.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2771333, 0.2875104
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2433867, 0.2539789
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2366219, 0.2385417
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3369522, 0.3366923
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2727621, 0.2616779
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.2999641, 0.3012008
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2811900, 0.2762297
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2305869, 0.2387233
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2885216, 0.2836900
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2998061, 0.2897447

Time for backsubstitution: 21.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6154

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6154

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2042646, upper bound: 0.2142803
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2036644, upper bound: 0.2148805
time: 2.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2832330, 0.2814088
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2526878, 0.2446790
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2393073, 0.2358558
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3363634, 0.3372831
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2637370, 0.2707063
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3005219, 0.3006430
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2774044, 0.2800167
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2379924, 0.2313187
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2857843, 0.2864271
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2918673, 0.2976839

Time for backsubstitution: 20.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6154

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 6154

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2132729, upper bound: 0.2052427
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2126727, upper bound: 0.2058429
time: 2.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2814089, 0.2832330
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2446790, 0.2526878
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2358558, 0.2393073
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3372831, 0.3363634
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2707064, 0.2637368
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3006430, 0.3005219
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2800165, 0.2774046
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2313186, 0.2379925
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2864271, 0.2857844
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2976838, 0.2918674

Time for backsubstitution: 21.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6154

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 6154

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2058429, upper bound: 0.2126724
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2052429, upper bound: 0.2132725
time: 2.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2875104, 0.2771333
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2539789, 0.2433867
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2385417, 0.2366219
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3366923, 0.3369521
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2616780, 0.2727621
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3012007, 0.2999641
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2762297, 0.2811900
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2387233, 0.2305870
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2836900, 0.2885216
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2897446, 0.2998062

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6154

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 6154

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2148803, upper bound: 0.2036640
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2142807, upper bound: 0.2042642
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2800908, 0.2845513
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2462897, 0.2510771
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2365662, 0.2385970
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3339100, 0.3397366
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2681522, 0.2662901
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3013505, 0.2998140
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2764884, 0.2809337
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2328491, 0.2364618
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2871557, 0.2850559
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2970500, 0.2925010

Time for backsubstitution: 21.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6154

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6154

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2073297, upper bound: 0.2112140
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2067295, upper bound: 0.2118141
time: 2.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2861911, 0.2784504
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2555895, 0.2417761
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2392522, 0.2359115
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3333191, 0.3403256
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2591233, 0.2753146
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.3019084, 0.2992563
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2726992, 0.2847168
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2402532, 0.2290558
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2844184, 0.2877933
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2891103, 0.3004392

Time for backsubstitution: 21.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6154

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6154

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163365, upper bound: 0.2021697
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2157369, upper bound: 0.2027700
time: 2.87 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 27.74 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 1, lower bound: -0.2027701, upper bound: 0.2157365
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 1, lower bound: -0.2021699, upper bound: 0.2163367
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 1, lower bound: -0.2118141, upper bound: 0.2067292
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 1, lower bound: -0.2112139, upper bound: 0.2073293
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 1, lower bound: -0.2042646, upper bound: 0.2142803
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 1, lower bound: -0.2036644, upper bound: 0.2148805
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 1, lower bound: -0.2132729, upper bound: 0.2052427
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 1, lower bound: -0.2126727, upper bound: 0.2058429
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 1, lower bound: -0.2058429, upper bound: 0.2126724
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 1, lower bound: -0.2052429, upper bound: 0.2132725
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 1, lower bound: -0.2148803, upper bound: 0.2036640
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 1, lower bound: -0.2142807, upper bound: 0.2042642
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 1, lower bound: -0.2073297, upper bound: 0.2112140
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 1, lower bound: -0.2067295, upper bound: 0.2118141
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 1, lower bound: -0.2163365, upper bound: 0.2021697
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.74
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

Time for backsubstitution: 21.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 2873
type: DSZ, layer: 3, pos: 1117
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1514
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1803

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 2390

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1975004, upper bound: 0.2123953
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1994289, upper bound: 0.2104668
time: 3.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2755311, 0.2861911
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2412365, 0.2555896
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2359115, 0.2381765
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3367691, 0.3333191
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2748226, 0.2591232
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.2992562, 0.3003756
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2847167, 0.2715960
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2290557, 0.2392272
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2864288, 0.2844185
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2980100, 0.2891104

Time for backsubstitution: 21.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 2873
type: DSZ, layer: 3, pos: 1117
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1514
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1803

Time for candidate selection: 0.51 seconds

### Candidate
type: DSZ, layer: 3, pos: 2390

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1969002, upper bound: 0.2129954
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1988288, upper bound: 0.2110670
time: 2.99 seconds

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

Time for backsubstitution: 21.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 2873
type: DSZ, layer: 3, pos: 1117
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1514
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1803

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 3, pos: 2390

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2065448, upper bound: 0.2033882
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2084732, upper bound: 0.2014594
time: 2.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2816321, 0.2800908
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2505376, 0.2462897
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2385970, 0.2354906
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3361802, 0.3339100
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2657980, 0.2681524
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.2998141, 0.2998178
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2809335, 0.2753850
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2364617, 0.2318231
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2836914, 0.2871556
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2900717, 0.2970500

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 2873
type: DSZ, layer: 3, pos: 1117
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1514
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1803

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 3, pos: 2390

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2059444, upper bound: 0.2039882
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2078731, upper bound: 0.2020596
time: 2.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 21.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 2873
type: DSZ, layer: 3, pos: 1117
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1514
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1803

Time for candidate selection: 0.37 seconds

### Candidate
type: DSZ, layer: 3, pos: 2390

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1989949, upper bound: 0.2109393
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2009232, upper bound: 0.2090105
time: 2.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.8527379, -7.1150637, -7.8527379, -7.1150637, -0.2742140, 0.2875104
1: 1.9682337, 2.6356406, 1.9682337, 2.6356406, -0.2428470, 0.2539789
2: -6.2961688, -5.7469501, -6.2961688, -5.7469501, -0.2366219, 0.2374661
3: -12.9902086, -12.1706266, -12.9902086, -12.1706266, -0.3333957, 0.3366923
4: -4.2042732, -3.6759524, -4.2042732, -3.6759524, -0.2722701, 0.2616779
5: -8.6626053, -7.9317474, -8.6626053, -7.9317474, -0.2999641, 0.2996681
6: -5.0156097, -4.4014921, -5.0156097, -4.4014921, -0.2811900, 0.2751265
7: -6.5901818, -6.0085917, -6.5901818, -6.0085917, -0.2305869, 0.2376973
8: -1.4708347, -0.8462844, -1.4708347, -0.8462844, -0.2871572, 0.2836900
9: -7.9338255, -7.1895881, -7.9338255, -7.1895881, -0.2973770, 0.2897447

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 2873
type: DSZ, layer: 3, pos: 1117
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1514
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1803

Time for candidate selection: 0.37 seconds

### Candidate
type: DSZ, layer: 3, pos: 2390

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1983947, upper bound: 0.2115395
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2003229, upper bound: 0.2096105
time: 2.78 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 27.63 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.63
Output dim: 1, lower bound: -0.1975004, upper bound: 0.2123953
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.63
Output dim: 1, lower bound: -0.1994289, upper bound: 0.2104668
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.63
Output dim: 1, lower bound: -0.1969002, upper bound: 0.2129954
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.63
Output dim: 1, lower bound: -0.1988288, upper bound: 0.2110670
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.63
Output dim: 1, lower bound: -0.2065448, upper bound: 0.2033882
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.63
Output dim: 1, lower bound: -0.2084732, upper bound: 0.2014594
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.63
Output dim: 1, lower bound: -0.2059444, upper bound: 0.2039882
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.63
Output dim: 1, lower bound: -0.2078731, upper bound: 0.2020596
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.63
Output dim: 1, lower bound: -0.1989949, upper bound: 0.2109393
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.63
Output dim: 1, lower bound: -0.2009232, upper bound: 0.2090105
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.63
Output dim: 1, lower bound: -0.1983947, upper bound: 0.2115395
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.63
Output dim: 1, lower bound: -0.2003229, upper bound: 0.2096105
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 1, lower bound: -0.2132729, upper bound: 0.2052427
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 1, lower bound: -0.2126727, upper bound: 0.2058429
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 1, lower bound: -0.2058429, upper bound: 0.2126724
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 1, lower bound: -0.2052429, upper bound: 0.2132725
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 1, lower bound: -0.2148803, upper bound: 0.2036640
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 1, lower bound: -0.2142807, upper bound: 0.2042642
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 1, lower bound: -0.2073297, upper bound: 0.2112140
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 1, lower bound: -0.2067295, upper bound: 0.2118141
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 1, lower bound: -0.2163365, upper bound: 0.2021697
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.63
Output dim: 1, lower bound: -0.2157369, upper bound: 0.2027700

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 53.77 + 547.97 = 601.74 seconds
