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
execution time: IAR + RelationalAnalysis = 22.93 + 33.26 = 56.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.2163591, upper bound: 0.2163587

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 6154
type: B, layer: 1, pos: 6154

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5788

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2072339, upper bound: 0.2163380
time: 2.75 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163491, upper bound: 0.2163507
time: 2.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.74 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.74
Output dim: 1, lower bound: -0.2072339, upper bound: 0.2163380
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.74
Output dim: 1, lower bound: -0.2163491, upper bound: 0.2163507

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -7.8434725, -7.1151276, -7.8507872, -7.1150780, -0.2901852, 0.2973722
1: 1.9814237, 2.6354170, 1.9710099, 2.6355958, -0.2553944, 0.2650828
2: -6.2910147, -5.7470036, -6.2950811, -5.7469616, -0.2356887, 0.2393669
3: -12.9902096, -12.1739550, -12.9902105, -12.1713314, -0.3472960, 0.3449115
4: -4.2036886, -3.6885448, -4.2041535, -3.6786027, -0.2900579, 0.2808883
5: -8.6618252, -7.9319377, -8.6624422, -7.9317870, -0.3057308, 0.3061804
6: -5.0155830, -4.4087467, -5.0156031, -4.4030232, -0.3001950, 0.2946323
7: -6.5789657, -6.0086064, -6.5878201, -6.0085936, -0.2400447, 0.2484822
8: -1.4701438, -0.8502569, -1.4706931, -0.8471208, -0.2903879, 0.2879484
9: -7.9333849, -7.2007275, -7.9337368, -7.1919341, -0.3078308, 0.3005750

Time for backsubstitution: 20.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 5802
type: B, layer: 1, pos: 6154
type: A, layer: 1, pos: 6154

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5788

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2072301, upper bound: 0.2072303
time: 2.78 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2072302, upper bound: 0.2163380
time: 2.83 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -7.8528347, -7.0991259, -7.8527350, -7.1150637, -0.2970803, 0.3074227
1: 1.9681801, 2.6599636, 1.9682405, 2.6356423, -0.2650862, 0.2752510
2: -6.2964287, -5.7396355, -6.2961683, -5.7469506, -0.2398214, 0.2456109
3: -12.9919138, -12.1701231, -12.9902086, -12.1706295, -0.3496476, 0.3482268
4: -4.2272778, -3.6759543, -4.2042737, -3.6759582, -0.3007214, 0.2905090
5: -8.6626539, -7.9304829, -8.6626043, -7.9317479, -0.3066481, 0.3076969
6: -5.0252995, -4.4011421, -5.0156102, -4.4014959, -0.3066489, 0.3006911
7: -6.5903201, -5.9890847, -6.5901771, -6.0085912, -0.2481439, 0.2539095
8: -1.4781294, -0.8462086, -1.4708319, -0.8462882, -0.2984295, 0.2909917
9: -7.9541779, -7.1895161, -7.9338260, -7.1895933, -0.3198885, 0.3092906

Time for backsubstitution: 21.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 6154
type: A, layer: 1, pos: 6154

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5788

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163385, upper bound: 0.2072303
time: 2.89 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163385, upper bound: 0.2072303
time: 2.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.51 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.51
Output dim: 1, lower bound: -0.2072301, upper bound: 0.2072303
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.51
Output dim: 1, lower bound: -0.2072302, upper bound: 0.2163380
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.51
Output dim: 1, lower bound: -0.2163385, upper bound: 0.2072303
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.51
Output dim: 1, lower bound: -0.2163385, upper bound: 0.2072303

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -7.8434725, -7.1151276, -7.8434725, -7.1151276, -0.2900470, 0.2900470
1: 1.9814237, 2.6354170, 1.9814237, 2.6354170, -0.2546482, 0.2546482
2: -6.2910147, -5.7470036, -6.2910147, -5.7470036, -0.2355917, 0.2355917
3: -12.9902096, -12.1739550, -12.9902096, -12.1739550, -0.3449109, 0.3449109
4: -4.2036886, -3.6885448, -4.2036886, -3.6885448, -0.2799782, 0.2799783
5: -8.6618252, -7.9319377, -8.6618252, -7.9319377, -0.3055142, 0.3055142
6: -5.0155830, -4.4087467, -5.0155830, -4.4087467, -0.2945399, 0.2945399
7: -6.5789657, -6.0086064, -6.5789657, -6.0086064, -0.2400305, 0.2400305
8: -1.4701438, -0.8502569, -1.4701438, -0.8502569, -0.2872463, 0.2872463
9: -7.9333849, -7.2007275, -7.9333849, -7.2007275, -0.2989525, 0.2989524

Time for backsubstitution: 21.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 6154
type: B, layer: 1, pos: 6154

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2041475, upper bound: 0.2072221
time: 2.87 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2072261, upper bound: 0.2072221
time: 2.93 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -7.8434725, -7.1151276, -7.8528347, -7.0991259, -0.2981365, 0.2994119
1: 1.9814237, 2.6354170, 1.9681801, 2.6599636, -0.2620236, 0.2679759
2: -6.2910147, -5.7470036, -6.2964287, -5.7396355, -0.2408149, 0.2408034
3: -12.9902096, -12.1739550, -12.9919138, -12.1701231, -0.3484756, 0.3466224
4: -4.2036886, -3.6885448, -4.2272778, -3.6759543, -0.2925763, 0.2879364
5: -8.6618252, -7.9319377, -8.6626539, -7.9304829, -0.3068526, 0.3063977
6: -5.0155830, -4.4087467, -5.0252995, -4.4011421, -0.3021417, 0.2994553
7: -6.5789657, -6.0086064, -6.5903201, -5.9890847, -0.2431922, 0.2508181
8: -1.4701438, -0.8502569, -1.4781294, -0.8462086, -0.2912256, 0.2944477
9: -7.9333849, -7.2007275, -7.9541779, -7.1895161, -0.3102314, 0.3086306

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 6154
type: B, layer: 1, pos: 6154

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2072266, upper bound: 0.2132640
time: 3.02 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2072266, upper bound: 0.2163296
time: 3.19 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -7.8528347, -7.0991259, -7.8434725, -7.1151276, -0.2994119, 0.2981365
1: 1.9681801, 2.6599636, 1.9814237, 2.6354170, -0.2679759, 0.2620236
2: -6.2964287, -5.7396355, -6.2910147, -5.7470036, -0.2408034, 0.2408149
3: -12.9919138, -12.1701231, -12.9902096, -12.1739550, -0.3466223, 0.3484756
4: -4.2272778, -3.6759543, -4.2036886, -3.6885448, -0.2879364, 0.2925763
5: -8.6626539, -7.9304829, -8.6618252, -7.9319377, -0.3063977, 0.3068526
6: -5.0252995, -4.4011421, -5.0155830, -4.4087467, -0.2994553, 0.3021417
7: -6.5903201, -5.9890847, -6.5789657, -6.0086064, -0.2508181, 0.2431921
8: -1.4781294, -0.8462086, -1.4701438, -0.8502569, -0.2944477, 0.2912257
9: -7.9541779, -7.1895161, -7.9333849, -7.2007275, -0.3086306, 0.3102314

Time for backsubstitution: 22.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 6154
type: A, layer: 1, pos: 6154

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2132624, upper bound: 0.2072222
time: 3.12 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163277, upper bound: 0.2072222
time: 3.07 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -7.8528347, -7.0991259, -7.8528347, -7.0991259, -0.2973545, 0.2973545
1: 1.9681801, 2.6599636, 1.9681801, 2.6599636, -0.2655438, 0.2655439
2: -6.2964287, -5.7396355, -6.2964287, -5.7396355, -0.2402359, 0.2402359
3: -12.9919138, -12.1701231, -12.9919138, -12.1701231, -0.3483976, 0.3483977
4: -4.2272778, -3.6759543, -4.2272778, -3.6759543, -0.2908059, 0.2908061
5: -8.6626539, -7.9304829, -8.6626539, -7.9304829, -0.3067739, 0.3067740
6: -5.0252995, -4.4011421, -5.0252995, -4.4011421, -0.3006971, 0.3006972
7: -6.5903201, -5.9890847, -6.5903201, -5.9890847, -0.2485951, 0.2485951
8: -1.4781294, -0.8462086, -1.4781294, -0.8462086, -0.2916793, 0.2916793
9: -7.9541779, -7.1895161, -7.9541779, -7.1895161, -0.3094512, 0.3094512

Time for backsubstitution: 21.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 6154
type: B, layer: 1, pos: 6154

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2132643, upper bound: 0.2072222
time: 2.88 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163297, upper bound: 0.2072220
time: 3.02 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 27.39 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.39
Output dim: 1, lower bound: -0.2041475, upper bound: 0.2072221
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.39
Output dim: 1, lower bound: -0.2072261, upper bound: 0.2072221
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 27.39
Output dim: 1, lower bound: -0.2072266, upper bound: 0.2132640
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 27.39
Output dim: 1, lower bound: -0.2072266, upper bound: 0.2163296
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.39
Output dim: 1, lower bound: -0.2132624, upper bound: 0.2072222
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.39
Output dim: 1, lower bound: -0.2163277, upper bound: 0.2072222
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.39
Output dim: 1, lower bound: -0.2132643, upper bound: 0.2072222
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.39
Output dim: 1, lower bound: -0.2163297, upper bound: 0.2072220

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.8360310, -7.1226711, -7.8408012, -7.1152697, -0.2824377, 0.2782864
1: 1.9892330, 2.6254468, 1.9841828, 2.6353049, -0.2464918, 0.2423742
2: -6.2905374, -5.7520885, -6.2908516, -5.7472687, -0.2348682, 0.2301206
3: -12.9859600, -12.1859951, -12.9902020, -12.1766233, -0.3370552, 0.3340383
4: -4.1933923, -3.7013688, -4.2035131, -3.6931508, -0.2651951, 0.2670254
5: -8.6568937, -7.9347978, -8.6602287, -7.9320469, -0.2996378, 0.3013002
6: -5.0077152, -4.4233665, -5.0154910, -4.4133749, -0.2822310, 0.2789473
7: -6.5727110, -6.0172176, -6.5767689, -6.0086994, -0.2335311, 0.2275439
8: -1.4643359, -0.8532834, -1.4698677, -0.8513370, -0.2785197, 0.2841450
9: -7.9272995, -7.2085991, -7.9332180, -7.2035074, -0.2904783, 0.2907051

Time for backsubstitution: 21.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 5802
type: A, layer: 1, pos: 6154
type: B, layer: 1, pos: 6154

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2041477, upper bound: 0.2041477
time: 2.73 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2041479, upper bound: 0.2072266
time: 2.74 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.8434534, -7.1151290, -7.8434682, -7.1151266, -0.2811136, 0.2900428
1: 1.9814335, 2.6354160, 1.9814264, 2.6354172, -0.2459209, 0.2546461
2: -6.2910137, -5.7470045, -6.2910137, -5.7470026, -0.2355690, 0.2357856
3: -12.9902105, -12.1739664, -12.9902105, -12.1739578, -0.3444805, 0.3389225
4: -4.2036881, -3.6885607, -4.2036886, -3.6885490, -0.2788045, 0.2661332
5: -8.6618128, -7.9319382, -8.6618223, -7.9319382, -0.3012978, 0.3055130
6: -5.0155811, -4.4087648, -5.0155830, -4.4087510, -0.2918051, 0.2804034
7: -6.5789566, -6.0086069, -6.5789647, -6.0086060, -0.2331983, 0.2387077
8: -1.4701438, -0.8502598, -1.4701428, -0.8502574, -0.2867832, 0.2850568
9: -7.9333849, -7.2007341, -7.9333863, -7.2007284, -0.2989495, 0.2906861

Time for backsubstitution: 21.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 6154
type: B, layer: 1, pos: 6154

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2072269, upper bound: 0.2041476
time: 2.79 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2072266, upper bound: 0.2072266
time: 2.81 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -7.8408012, -7.1152697, -7.8453898, -7.1067233, -0.2846178, 0.2918027
1: 1.9841828, 2.6353049, 1.9759909, 2.6499205, -0.2476449, 0.2598218
2: -6.2908516, -5.7472687, -6.2959437, -5.7447915, -0.2352163, 0.2400637
3: -12.9902020, -12.1766233, -12.9876652, -12.1820917, -0.3376738, 0.3387688
4: -4.2035131, -3.6931508, -4.2169456, -3.6887736, -0.2796246, 0.2700751
5: -8.6602287, -7.9320469, -8.6577320, -7.9333415, -0.3026381, 0.3005478
6: -5.0154910, -4.4133749, -5.0174389, -4.4157391, -0.2866098, 0.2842533
7: -6.5767689, -6.0086994, -6.5840626, -5.9977751, -0.2292551, 0.2442843
8: -1.4698677, -0.8513370, -1.4722672, -0.8492351, -0.2881205, 0.2853433
9: -7.9332180, -7.2035074, -7.9480801, -7.1973886, -0.3019868, 0.2983469

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 5802
type: B, layer: 1, pos: 6154
type: A, layer: 1, pos: 6154

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2041475, upper bound: 0.2132620
time: 2.84 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2041476, upper bound: 0.2041442
time: 5.25 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -7.8434682, -7.1151266, -7.8528175, -7.0991268, -0.2932881, 0.2904823
1: 1.9814264, 2.6354172, 1.9681921, 2.6599629, -0.2568997, 0.2592514
2: -6.2910137, -5.7470026, -6.2964272, -5.7396355, -0.2404216, 0.2407806
3: -12.9902105, -12.1739578, -12.9919119, -12.1701374, -0.3425857, 0.3461922
4: -4.2036886, -3.6885490, -4.2272787, -3.6759734, -0.2787349, 0.2798179
5: -8.6618223, -7.9319382, -8.6626444, -7.9304848, -0.3068517, 0.3021822
6: -5.0155830, -4.4087510, -5.0253000, -4.4011621, -0.2880142, 0.2918389
7: -6.5789647, -6.0086060, -6.5903072, -5.9890842, -0.2391207, 0.2437959
8: -1.4701428, -0.8502574, -1.4781294, -0.8462114, -0.2890313, 0.2923446
9: -7.9333863, -7.2007284, -7.9541759, -7.1895261, -0.3019671, 0.3040668

Time for backsubstitution: 21.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 6154
type: A, layer: 1, pos: 6154

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2041476, upper bound: 0.2163298
time: 2.92 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2041476, upper bound: 0.2163278
time: 2.88 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.8453898, -7.1067233, -7.8408012, -7.1152697, -0.2918026, 0.2846178
1: 1.9759909, 2.6499205, 1.9841828, 2.6353049, -0.2598218, 0.2476448
2: -6.2959437, -5.7447915, -6.2908516, -5.7472687, -0.2400638, 0.2352162
3: -12.9876652, -12.1820917, -12.9902020, -12.1766233, -0.3387688, 0.3376738
4: -4.2169456, -3.6887736, -4.2035131, -3.6931508, -0.2700751, 0.2796246
5: -8.6577320, -7.9333415, -8.6602287, -7.9320469, -0.3005478, 0.3026381
6: -5.0174389, -4.4157391, -5.0154910, -4.4133749, -0.2842534, 0.2866098
7: -6.5840626, -5.9977751, -6.5767689, -6.0086994, -0.2442843, 0.2292551
8: -1.4722672, -0.8492351, -1.4698677, -0.8513370, -0.2853432, 0.2881205
9: -7.9480801, -7.1973886, -7.9332180, -7.2035074, -0.2983470, 0.3019868

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 5802
type: A, layer: 1, pos: 6154
type: B, layer: 1, pos: 6154

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2132621, upper bound: 0.2041475
time: 2.81 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2132622, upper bound: 0.2072265
time: 2.80 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.8528175, -7.0991268, -7.8434682, -7.1151266, -0.2904823, 0.2932881
1: 1.9681921, 2.6599629, 1.9814264, 2.6354172, -0.2592514, 0.2568997
2: -6.2964272, -5.7396355, -6.2910137, -5.7470026, -0.2407806, 0.2404216
3: -12.9919119, -12.1701374, -12.9902105, -12.1739578, -0.3461922, 0.3425857
4: -4.2272787, -3.6759734, -4.2036886, -3.6885490, -0.2798178, 0.2787349
5: -8.6626444, -7.9304848, -8.6618223, -7.9319382, -0.3021822, 0.3068517
6: -5.0253000, -4.4011621, -5.0155830, -4.4087510, -0.2918390, 0.2880141
7: -6.5903072, -5.9890842, -6.5789647, -6.0086060, -0.2437959, 0.2391207
8: -1.4781294, -0.8462114, -1.4701428, -0.8502574, -0.2923446, 0.2890314
9: -7.9541759, -7.1895261, -7.9333863, -7.2007284, -0.3040668, 0.3019671

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 6154
type: B, layer: 1, pos: 6154

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163283, upper bound: 0.2041476
time: 2.86 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163282, upper bound: 0.2072265
time: 3.10 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.8453898, -7.1067233, -7.8501616, -7.0992694, -0.2897420, 0.2855967
1: 1.9759909, 2.6499205, 1.9709415, 2.6598473, -0.2573850, 0.2530820
2: -6.2959437, -5.7447915, -6.2962627, -5.7399025, -0.2394890, 0.2346655
3: -12.9876652, -12.1820917, -12.9919071, -12.1727848, -0.3405273, 0.3375959
4: -4.2169456, -3.6887736, -4.2270942, -3.6805587, -0.2760065, 0.2778436
5: -8.6577320, -7.9333415, -8.6610622, -7.9305925, -0.3009094, 0.3025632
6: -5.0174389, -4.4157391, -5.0252066, -4.4057684, -0.2883826, 0.2851663
7: -6.5840626, -5.9977751, -6.5881219, -5.9891815, -0.2420890, 0.2360560
8: -1.4722672, -0.8492351, -1.4778552, -0.8472881, -0.2829577, 0.2885618
9: -7.9480801, -7.1973886, -7.9540110, -7.1922951, -0.3009636, 0.3012018

Time for backsubstitution: 22.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 5802
type: A, layer: 1, pos: 6154
type: B, layer: 1, pos: 6154

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2132781, upper bound: 0.2042699
time: 3.20 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2132785, upper bound: 0.2073351
time: 2.96 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.8528175, -7.0991268, -7.8528318, -7.0991259, -0.2884245, 0.2973498
1: 1.9681921, 2.6599629, 1.9681838, 2.6599636, -0.2568156, 0.2655419
2: -6.2964272, -5.7396355, -6.2964268, -5.7396336, -0.2402132, 0.2404293
3: -12.9919119, -12.1701374, -12.9919119, -12.1701269, -0.3479671, 0.3425079
4: -4.2272787, -3.6759734, -4.2272778, -3.6759586, -0.2906605, 0.2769580
5: -8.6626444, -7.9304848, -8.6626530, -7.9304843, -0.3025578, 0.3067728
6: -5.0253000, -4.4011621, -5.0252991, -4.4011455, -0.2983640, 0.2865664
7: -6.5903072, -5.9890842, -6.5903168, -5.9890842, -0.2417672, 0.2478454
8: -1.4781294, -0.8462114, -1.4781299, -0.8462100, -0.2912391, 0.2894461
9: -7.9541759, -7.1895261, -7.9541759, -7.1895180, -0.3094485, 0.3011848

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 6154
type: B, layer: 1, pos: 6154

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163408, upper bound: 0.2042699
time: 2.85 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2163403, upper bound: 0.2073349
time: 2.95 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.24 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2041477, upper bound: 0.2041477
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2041479, upper bound: 0.2072266
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2072269, upper bound: 0.2041476
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2072266, upper bound: 0.2072266
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2041475, upper bound: 0.2132620
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2041476, upper bound: 0.2041442
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2041476, upper bound: 0.2163298
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2041476, upper bound: 0.2163278
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2132621, upper bound: 0.2041475
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2132622, upper bound: 0.2072265
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2163283, upper bound: 0.2041476
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2163282, upper bound: 0.2072265
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2132781, upper bound: 0.2042699
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2132785, upper bound: 0.2073351
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2163408, upper bound: 0.2042699
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.24
Output dim: 1, lower bound: -0.2163403, upper bound: 0.2073349

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.8360310, -7.1226711, -7.8360310, -7.1226711, -0.2734991, 0.2734991
1: 1.9892330, 2.6254468, 1.9892330, 2.6254468, -0.2371325, 0.2371326
2: -6.2905374, -5.7520885, -6.2905374, -5.7520885, -0.2298504, 0.2298503
3: -12.9859600, -12.1859951, -12.9859600, -12.1859951, -0.3293592, 0.3293592
4: -4.1933923, -3.7013688, -4.1933923, -3.7013688, -0.2569706, 0.2569708
5: -8.6568937, -7.9347978, -8.6568937, -7.9347978, -0.2970006, 0.2970006
6: -5.0077152, -4.4233665, -5.0077152, -4.4233665, -0.2712837, 0.2712837
7: -6.5727110, -6.0172176, -6.5727110, -6.0172176, -0.2233102, 0.2233103
8: -1.4643359, -0.8532834, -1.4643359, -0.8532834, -0.2770230, 0.2770230
9: -7.9272995, -7.2085991, -7.9272995, -7.2085991, -0.2852191, 0.2852191

Time for backsubstitution: 21.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 6154
type: B, layer: 1, pos: 6154

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 5802

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2025942, upper bound: 0.2041410
time: 2.87 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2041448, upper bound: 0.2041478
time: 2.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.8360310, -7.1226711, -7.8434534, -7.1151290, -0.2825754, 0.2809563
1: 1.9892330, 2.6254468, 1.9814335, 2.6354160, -0.2465696, 0.2452038
2: -6.2905374, -5.7520885, -6.2910137, -5.7470045, -0.2350138, 0.2303394
3: -12.9859600, -12.1859951, -12.9902105, -12.1739664, -0.3402129, 0.3336174
4: -4.1933923, -3.7013688, -4.2036881, -3.6885607, -0.2690922, 0.2659470
5: -8.6568937, -7.9347978, -8.6618128, -7.9319382, -0.2996606, 0.3028494
6: -5.0077152, -4.4233665, -5.0155811, -4.4087648, -0.2844671, 0.2761917
7: -6.5727110, -6.0172176, -6.5789566, -6.0086069, -0.2321997, 0.2289934
8: -1.4643359, -0.8532834, -1.4701438, -0.8502598, -0.2797370, 0.2840652
9: -7.9272995, -7.2085991, -7.9333849, -7.2007341, -0.2932938, 0.2908680

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 5802
type: A, layer: 1, pos: 6154
type: B, layer: 1, pos: 6154

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5802

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2041383, upper bound: 0.2056735
time: 2.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2041452, upper bound: 0.2072238
time: 3.12 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.8434534, -7.1151290, -7.8360310, -7.1226711, -0.2809563, 0.2825753
1: 1.9814335, 2.6354160, 1.9892330, 2.6254468, -0.2452039, 0.2465696
2: -6.2910137, -5.7470045, -6.2905374, -5.7520885, -0.2303395, 0.2350138
3: -12.9902105, -12.1739664, -12.9859600, -12.1859951, -0.3336174, 0.3402131
4: -4.2036881, -3.6885607, -4.1933923, -3.7013688, -0.2659471, 0.2690922
5: -8.6618128, -7.9319382, -8.6568937, -7.9347978, -0.3028495, 0.2996606
6: -5.0155811, -4.4087648, -5.0077152, -4.4233665, -0.2761917, 0.2844672
7: -6.5789566, -6.0086069, -6.5727110, -6.0172176, -0.2289934, 0.2321996
8: -1.4701438, -0.8502598, -1.4643359, -0.8532834, -0.2840652, 0.2797370
9: -7.9333849, -7.2007341, -7.9272995, -7.2085991, -0.2908680, 0.2932938

Time for backsubstitution: 21.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 5802
type: B, layer: 1, pos: 6154
type: A, layer: 1, pos: 6154

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5802

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2056728, upper bound: 0.2041384
time: 2.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2072231, upper bound: 0.2041451
time: 3.10 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.8434534, -7.1151290, -7.8434534, -7.1151290, -0.2811131, 0.2811131
1: 1.9814335, 2.6354160, 1.9814335, 2.6354160, -0.2459207, 0.2459207
2: -6.2910137, -5.7470045, -6.2910137, -5.7470045, -0.2357847, 0.2357846
3: -12.9902105, -12.1739664, -12.9902105, -12.1739664, -0.3389219, 0.3389219
4: -4.2036881, -3.6885607, -4.2036881, -3.6885607, -0.2661331, 0.2661331
5: -8.6618128, -7.9319382, -8.6618128, -7.9319382, -0.3012975, 0.3012975
6: -5.0155811, -4.4087648, -5.0155811, -4.4087648, -0.2804034, 0.2804034
7: -6.5789566, -6.0086069, -6.5789566, -6.0086069, -0.2331983, 0.2331982
8: -1.4701438, -0.8502598, -1.4701438, -0.8502598, -0.2850548, 0.2850548
9: -7.9333849, -7.2007341, -7.9333849, -7.2007341, -0.2906857, 0.2906857

Time for backsubstitution: 22.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 6154
type: B, layer: 1, pos: 6154

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5802

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2056736, upper bound: 0.2041383
time: 2.86 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2072238, upper bound: 0.2041451
time: 2.90 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -7.8360310, -7.1226711, -7.8453898, -7.1067233, -0.2815557, 0.2828641
1: 1.9892330, 2.6254468, 1.9759909, 2.6499205, -0.2441527, 0.2504626
2: -6.2905374, -5.7520885, -6.2959437, -5.7447915, -0.2349591, 0.2350459
3: -12.9859600, -12.1859951, -12.9876652, -12.1820917, -0.3329948, 0.3310728
4: -4.1933923, -3.7013688, -4.2169456, -3.6887736, -0.2695700, 0.2648405
5: -8.6568937, -7.9347978, -8.6577320, -7.9333415, -0.2983387, 0.2979107
6: -5.0077152, -4.4233665, -5.0174389, -4.4157391, -0.2789463, 0.2761429
7: -6.5727110, -6.0172176, -6.5840626, -5.9977751, -0.2263781, 0.2340288
8: -1.4643359, -0.8532834, -1.4722672, -0.8492351, -0.2809985, 0.2842090
9: -7.9272995, -7.2085991, -7.9480801, -7.1973886, -0.2965008, 0.2948303

Time for backsubstitution: 21.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5802
type: B, layer: 1, pos: 5802
type: A, layer: 1, pos: 6154
type: B, layer: 1, pos: 6154

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5802

## Relational analysis of NS_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2025967, upper bound: 0.2132592
time: 2.87 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2041473, upper bound: 0.2132612
time: 2.85 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -7.8434534, -7.1151290, -7.8453898, -7.1067233, -0.2846524, 0.2919404
1: 1.9814335, 2.6354160, 1.9759909, 2.6499205, -0.2477887, 0.2571548
2: -6.2910137, -5.7470045, -6.2959437, -5.7447915, -0.2352875, 0.2402093
3: -12.9902105, -12.1739664, -12.9876652, -12.1820917, -0.3372530, 0.3419267
4: -4.2036881, -3.6885607, -4.2169456, -3.6887736, -0.2748933, 0.2700893
5: -8.6618128, -7.9319382, -8.6577320, -7.9333415, -0.3041874, 0.3005707
6: -5.0155811, -4.4087648, -5.0174389, -4.4157391, -0.2821815, 0.2844984
7: -6.5789566, -6.0086069, -6.5840626, -5.9977751, -0.2294072, 0.2402641
8: -1.4701438, -0.8502598, -1.4722672, -0.8492351, -0.2880406, 0.2853473
9: -7.9333849, -7.2007341, -7.9480801, -7.1973886, -0.3021498, 0.2984817

Time for backsubstitution: 21.86 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.19 + 557.81 = 614.01 seconds
