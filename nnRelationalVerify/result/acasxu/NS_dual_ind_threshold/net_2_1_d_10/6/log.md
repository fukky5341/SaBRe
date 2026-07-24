## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 4974.543682465716


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2612.8769531, 3109.3488770, -2612.8769531, 3109.3488770, -5722.2255859, 5722.2255859)
1: (-2202.3388672, 2487.5476074, -2202.3388672, 2487.5476074, -4689.8857422, 4689.8857422)
2: (-1789.5975342, 2459.0048828, -1789.5975342, 2459.0048828, -4248.6025391, 4248.6025391)
3: (-2395.7185059, 3112.4526367, -2395.7185059, 3112.4526367, -5508.1708984, 5508.1708984)
4: (-2296.5754395, 3353.3630371, -2296.5754395, 3353.3630371, -5649.9384766, 5649.9384766)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.42 + 3.21 = 4.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -4974.5934284, upper bound: 4974.5934284

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5843863, upper bound: 4974.5875316
time: 1.25 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5841473, upper bound: 4974.5841473
time: 1.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.73 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.73
Output dim: 3, lower bound: -4974.5843863, upper bound: 4974.5875316
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.73
Output dim: 3, lower bound: -4974.5841473, upper bound: 4974.5841473

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2566.8178711, 3053.7199707, -2590.0441895, 3081.8823242, -5648.7001953, 5643.7641602
1: -2163.3325195, 2443.1081543, -2183.0468750, 2465.6228027, -4628.9550781, 4626.1552734
2: -1757.8508301, 2414.8166504, -1773.8958740, 2437.2153320, -4195.0664062, 4188.7119141
3: -2353.0764160, 3056.9091797, -2374.6550293, 3085.0336914, -5438.1103516, 5431.5644531
4: -2255.6923828, 3292.8603516, -2276.3640137, 3323.5158691, -5579.2080078, 5569.2236328

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5828931, upper bound: 4974.5857813
time: 1.22 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5783170, upper bound: 4974.5845517
time: 1.38 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2920.1215820, 3440.8747559, -2536.4248047, 3017.5434570, -5937.6650391, 5977.2988281
1: -2448.9824219, 2755.3217773, -2137.9592285, 2414.2167969, -4863.1987305, 4893.2802734
2: -1993.2526855, 2716.9411621, -1737.1340332, 2386.1481934, -4379.3652344, 4454.0751953
3: -2659.2255859, 3447.2041016, -2325.4282227, 3020.8796387, -5680.1054688, 5772.6323242
4: -2555.4533691, 3702.5310059, -2229.0493164, 3253.9118652, -5809.3652344, 5931.5795898

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5839064, upper bound: 4974.5828935
time: 1.44 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5828935, upper bound: 4974.5828935
time: 1.13 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.97 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.97
Output dim: 3, lower bound: -4974.5828931, upper bound: 4974.5857813
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.97
Output dim: 3, lower bound: -4974.5783170, upper bound: 4974.5845517
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.97
Output dim: 3, lower bound: -4974.5839064, upper bound: 4974.5828935
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.97
Output dim: 3, lower bound: -4974.5828935, upper bound: 4974.5828935

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -2526.0173340, 3004.8171387, -2526.2404785, 3005.0305176, -5531.0478516, 5531.0576172
1: -2128.3579102, 2404.0759277, -2128.3388672, 2404.2731934, -4532.6308594, 4532.4140625
2: -1729.7163086, 2376.0952148, -1729.8405762, 2376.3078613, -4106.0234375, 4105.9350586
3: -2315.4521484, 3007.7368164, -2315.6350098, 3007.7668457, -5323.2187500, 5323.3720703
4: -2219.7626953, 3239.7976074, -2220.0124512, 3240.0798340, -5459.8427734, 5459.8100586

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5766442, upper bound: 4974.5790855
time: 1.28 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5765641, upper bound: 4974.5784576
time: 1.13 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2543.4985352, 3025.2829590, -2716.7277832, 3238.5083008, -5782.0068359, 5742.0107422
1: -2143.4025879, 2420.1364746, -2291.2983398, 2590.6088867, -4734.0117188, 4711.4345703
2: -1741.6999512, 2392.2202148, -1861.7052002, 2562.3352051, -4304.0346680, 4253.9252930
3: -2331.1801758, 3028.0119629, -2493.8139648, 3241.5935059, -5572.7734375, 5521.8261719
4: -2234.8117676, 3261.8964844, -2389.8640137, 3494.8574219, -5729.6689453, 5651.7597656

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5782967, upper bound: 4974.5845249
time: 1.34 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5783170, upper bound: 4974.5845517
time: 1.09 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -2844.1113281, 3349.0224609, -2436.3139648, 2898.5161133, -5742.6274414, 5785.3359375
1: -2385.0500488, 2682.3593750, -2054.5598145, 2319.6372070, -4704.6860352, 4736.9184570
2: -1941.2615967, 2644.5793457, -1669.0541992, 2292.5581055, -4233.7602539, 4313.6337891
3: -2589.1403809, 3355.6704102, -2234.3745117, 2902.4636230, -5491.6040039, 5590.0449219
4: -2488.5000000, 3603.9741211, -2141.6975098, 3126.7700195, -5615.2700195, 5745.6718750

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5733822, upper bound: 4974.5687487
time: 1.37 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5815810, upper bound: 4974.5801121
time: 1.23 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -2881.1079102, 3393.3869629, -2489.8635254, 2960.8989258, -5842.0068359, 5883.2504883
1: -2415.9602051, 2717.3356934, -2099.0495605, 2368.5859375, -4784.5458984, 4816.3852539
2: -1966.3665771, 2679.0407715, -1705.3837891, 2341.1359863, -4307.4863281, 4384.4233398
3: -2622.7377930, 3399.8964844, -2281.6411133, 2964.3576660, -5587.0957031, 5681.5375977
4: -2520.5541992, 3650.4641113, -2187.3806152, 3191.9746094, -5712.5288086, 5837.8437500

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5733827, upper bound: 4974.5687631
time: 1.52 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5801121, upper bound: 4974.5801121
time: 1.54 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.46 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -4974.5766442, upper bound: 4974.5790855
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -4974.5765641, upper bound: 4974.5784576
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -4974.5782967, upper bound: 4974.5845249
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -4974.5783170, upper bound: 4974.5845517
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -4974.5733822, upper bound: 4974.5687487
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -4974.5815810, upper bound: 4974.5801121
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -4974.5733827, upper bound: 4974.5687631
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -4974.5801121, upper bound: 4974.5801121

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2416.7084961, 2874.6437988, -2450.5881348, 2914.8564453, -5331.5649414, 5325.2314453
1: -2037.2792969, 2300.5988770, -2065.2424316, 2332.6101074, -4369.8886719, 4365.8413086
2: -1655.3807373, 2273.6413574, -1678.3739014, 2305.3215332, -3960.7021484, 3952.0151367
3: -2215.9584961, 2878.0219727, -2246.7460938, 2917.8752441, -5133.8339844, 5124.7670898
4: -2124.2841797, 3100.5537109, -2153.9040527, 3143.6000977, -5267.8842773, 5254.4580078

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5766442, upper bound: 4974.5790855
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5766442, upper bound: 4974.5790855
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2478.2707520, 2946.9025879, -2484.3706055, 2953.6030273, -5431.8740234, 5431.2734375
1: -2088.3476562, 2357.3598633, -2092.4692383, 2362.8940430, -4451.2416992, 4449.8291016
2: -1697.1242676, 2330.0187988, -1700.7678223, 2335.0830078, -4032.2072754, 4030.7866211
3: -2270.6079102, 2949.8041992, -2275.9868164, 2956.4631348, -5227.0708008, 5225.7910156
4: -2177.0559082, 3176.4594727, -2182.2768555, 3183.3837891, -5360.4394531, 5358.7363281

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5765641, upper bound: 4974.5784576
time: 1.78 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5765641, upper bound: 4974.5784576
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2496.7409668, 2969.5354004, -2672.2536621, 3182.8608398, -5679.6015625, 5641.7885742
1: -2103.4426270, 2375.3032227, -2253.0512695, 2546.1298828, -4649.5722656, 4628.3535156
2: -1709.4123535, 2347.8789062, -1830.5760498, 2517.9362793, -4227.3486328, 4178.4550781
3: -2287.9533691, 2971.9309082, -2451.4514160, 3185.9912109, -5473.9443359, 5423.3823242
4: -2193.3403320, 3200.8522949, -2349.5329590, 3433.9174805, -5627.2578125, 5550.3852539

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5782967, upper bound: 4974.5845249
time: 1.26 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5782967, upper bound: 4974.5845249
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2502.4138184, 2973.9274902, -2664.2495117, 3176.9628906, -5679.3769531, 5638.1767578
1: -2108.9882812, 2379.6196289, -2247.6569824, 2541.5832520, -4650.5712891, 4627.2753906
2: -1713.2132568, 2351.7902832, -1826.0504150, 2514.0329590, -4227.2456055, 4177.8403320
3: -2292.5886230, 2977.1772461, -2446.3872070, 3180.0136719, -5472.6025391, 5423.5644531
4: -2197.6657715, 3206.2673340, -2344.1132812, 3429.0432129, -5626.7084961, 5550.3789062

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5783170, upper bound: 4974.5845517
time: 1.24 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5783170, upper bound: 4974.5845517
time: 1.46 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2787.3503418, 3282.3742676, -2389.5578613, 2842.4318848, -5629.4462891, 5671.9316406
1: -2336.9863281, 2628.9047852, -2014.6251221, 2274.6975098, -4611.6835938, 4643.5292969
2: -1902.2661133, 2591.7443848, -1636.7747803, 2247.9716797, -4149.5585938, 4228.5190430
3: -2537.0861816, 3288.7985840, -2190.9326172, 2846.2441406, -5383.3300781, 5479.7314453
4: -2438.5478516, 3531.8696289, -2100.2456055, 3065.8383789, -5504.3862305, 5632.1152344

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5712216, upper bound: 4974.5672465
time: 1.25 seconds

## Relational analysis of NS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5700288, upper bound: 4974.5651418
time: 1.19 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5700288, upper bound: 4974.5655689
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2785.3505859, 3281.3688965, -2400.1267090, 2857.6313477, -5642.9819336, 5681.4956055
1: -2335.9763184, 2627.8305664, -2024.5529785, 2286.8247070, -4622.8007812, 4652.3833008
2: -1901.2237549, 2591.4655762, -1644.5887451, 2260.5615234, -4161.7851562, 4236.0541992
3: -2535.9365234, 3287.4992676, -2202.0881348, 2861.3588867, -5397.2954102, 5489.5874023
4: -2437.2023926, 3531.9931641, -2110.5869141, 3083.4638672, -5520.6660156, 5642.5800781

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5795403, upper bound: 4974.5757370
time: 1.27 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5699440, upper bound: 4974.5718412
time: 1.33 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2830.7736816, 3334.5349121, -2444.9079590, 2907.1396484, -5737.4746094, 5779.4428711
1: -2373.4304199, 2670.0998535, -2060.4023438, 2325.3837891, -4698.8144531, 4730.5019531
2: -1931.8225098, 2632.3884277, -1674.1853027, 2298.2160645, -4229.4653320, 4306.5732422
3: -2576.7373047, 3340.8666992, -2239.9477539, 2910.3620605, -5487.0996094, 5580.8144531
4: -2476.3359375, 3586.8361816, -2147.4648438, 3133.4902344, -5609.8251953, 5734.3007812

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5722733, upper bound: 4974.5668396
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5722733, upper bound: 4974.5668396
time: 1.23 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2819.7187500, 3322.6394043, -2447.0207520, 2912.0498047, -5731.7675781, 5769.6601562
1: -2364.7963867, 2660.4179688, -2063.6843262, 2329.5393066, -4694.3354492, 4724.1020508
2: -1924.5653076, 2623.5629883, -1676.4982910, 2303.0688477, -4227.6342773, 4300.0615234
3: -2567.1086426, 3328.7214355, -2243.0988770, 2915.4006348, -5482.5092773, 5571.8193359
4: -2466.9113770, 3575.1916504, -2150.3784180, 3140.1608887, -5607.0712891, 5725.5703125

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5801121, upper bound: 4974.5801121
time: 1.15 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5801121, upper bound: 4974.5801121
time: 1.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.99 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5766442, upper bound: 4974.5790855
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5766442, upper bound: 4974.5790855
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5765641, upper bound: 4974.5784576
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5765641, upper bound: 4974.5784576
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5782967, upper bound: 4974.5845249
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5782967, upper bound: 4974.5845249
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5783170, upper bound: 4974.5845517
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5783170, upper bound: 4974.5845517
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5700288, upper bound: 4974.5651418
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5700288, upper bound: 4974.5655689
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5795403, upper bound: 4974.5757370
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5699440, upper bound: 4974.5718412
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5722733, upper bound: 4974.5668396
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5722733, upper bound: 4974.5668396
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5801121, upper bound: 4974.5801121
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 3, lower bound: -4974.5801121, upper bound: 4974.5801121

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2416.7084961, 2874.6437988, -2426.2524414, 2885.2795410, -5301.9877930, 5300.8950195
1: -2037.2792969, 2300.5988770, -2044.5653076, 2308.9794922, -4346.2583008, 4345.1625977
2: -1655.3807373, 2273.6413574, -1661.5556641, 2281.8144531, -3937.1950684, 3935.1965332
3: -2215.9584961, 2878.0219727, -2224.1022949, 2888.3232422, -5104.2817383, 5102.1235352
4: -2124.2841797, 3100.5537109, -2132.2182617, 3111.3901367, -5235.6738281, 5232.7714844

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5766162, upper bound: 4974.5790855
time: 1.27 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5604101, upper bound: 4974.5572121
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2416.7084961, 2874.6437988, -2775.8698730, 3266.5021973, -5683.2104492, 5650.5131836
1: -2037.2792969, 2300.5988770, -2326.0895996, 2616.3757324, -4653.6542969, 4626.6865234
2: -1655.3807373, 2273.6413574, -1894.0406494, 2579.0485840, -4234.4282227, 4167.6279297
3: -2215.9584961, 2878.0219727, -2525.7966309, 3272.5161133, -5488.4746094, 5403.8173828
4: -2124.2841797, 3100.5537109, -2428.0776367, 3514.1425781, -5638.4267578, 5528.6313477

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5766162, upper bound: 4974.5790855
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5604101, upper bound: 4974.5572121
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2478.2707520, 2946.9025879, -2462.0461426, 2926.4787598, -5404.7490234, 5408.9487305
1: -2088.3476562, 2357.3598633, -2073.4633789, 2341.2114258, -4429.5590820, 4430.8232422
2: -1697.1242676, 2330.0187988, -1685.3328857, 2313.5095215, -4010.6337891, 4015.3515625
3: -2270.6079102, 2949.8041992, -2255.2031250, 2929.3745117, -5199.9824219, 5205.0073242
4: -2177.0559082, 3176.4594727, -2162.3854980, 3153.8186035, -5330.8745117, 5338.8447266

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5765139, upper bound: 4974.5784576
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5765641, upper bound: 4974.5784576
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5763282, upper bound: 4974.5773061
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2478.2707520, 2946.9025879, -2812.6755371, 3310.4636230, -5788.7343750, 5759.5781250
1: -2088.3476562, 2357.3598633, -2356.7917480, 2651.0700684, -4739.4179688, 4714.1513672
2: -1697.1242676, 2330.0187988, -1918.9658203, 2613.2131348, -4310.3374023, 4248.9843750
3: -2270.6079102, 2949.8041992, -2559.1279297, 3316.3229980, -5586.9306641, 5508.9321289
4: -2177.0559082, 3176.4594727, -2459.9096680, 3560.1945801, -5737.2500000, 5636.3691406

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5765139, upper bound: 4974.5784576
time: 1.40 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5765641, upper bound: 4974.5784576
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5763282, upper bound: 4974.5773061
time: 1.27 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2496.7409668, 2969.5354004, -2644.8632812, 3149.4699707, -5646.2109375, 5614.3974609
1: -2103.4426270, 2375.3032227, -2229.8159180, 2519.4599609, -4622.9023438, 4605.1191406
2: -1709.4123535, 2347.8789062, -1811.6459961, 2491.4099121, -4200.8222656, 4159.5249023
3: -2287.9533691, 2971.9309082, -2425.9567871, 3152.6254883, -5440.5791016, 5397.8876953
4: -2193.3403320, 3200.8522949, -2325.0842285, 3397.5405273, -5590.8808594, 5525.9365234

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5613301, upper bound: 4974.5697323
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
time: 1.30 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2496.7409668, 2969.5354004, -2991.9172363, 3527.9145508, -6024.6552734, 5961.4521484
1: -2103.4426270, 2375.3032227, -2510.4887695, 2824.3581543, -4927.8007812, 4885.7919922
2: -1709.4123535, 2347.8789062, -2042.7139893, 2786.4045410, -4495.8168945, 4390.1718750
3: -2287.9533691, 2971.9309082, -2725.6274414, 3533.9438477, -5821.8974609, 5697.5585938
4: -2193.3403320, 3200.8522949, -2618.9055176, 3797.6738281, -5991.0141602, 5819.7578125

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5613301, upper bound: 4974.5697323
time: 1.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2502.4138184, 2973.9274902, -2636.7553711, 3143.5700684, -5645.9833984, 5610.6826172
1: -2108.9882812, 2379.6196289, -2224.3679199, 2514.8784180, -4623.8666992, 4603.9868164
2: -1713.2132568, 2351.7902832, -1807.0718994, 2487.4973145, -4200.7104492, 4158.8613281
3: -2292.5886230, 2977.1772461, -2420.8662109, 3146.6013184, -5439.1899414, 5398.0434570
4: -2197.6657715, 3206.2673340, -2319.6101074, 3392.6545410, -5590.3198242, 5525.8774414

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611489, upper bound: 4974.5693315
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611048, upper bound: 4974.5676510
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2502.4138184, 2973.9274902, -2969.5371094, 3503.6733398, -6006.0869141, 5943.4648438
1: -2108.9882812, 2379.6196289, -2491.9418945, 2804.7863770, -4913.7739258, 4871.5600586
2: -1713.2132568, 2351.7902832, -2027.8519287, 2767.6103516, -4480.8237305, 4379.6420898
3: -2292.5886230, 2977.1772461, -2706.3984375, 3509.6516113, -5802.2402344, 5683.5747070
4: -2197.6657715, 3206.2673340, -2600.2695312, 3772.3139648, -5969.9794922, 5806.5361328

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611489, upper bound: 4974.5693315
time: 1.21 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611048, upper bound: 4974.5676510
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2781.0654297, 3274.7873535, -2356.6804199, 2802.0039062, -5582.7475586, 5631.4663086
1: -2331.6555176, 2622.8374023, -1986.4787598, 2242.4196777, -4574.0751953, 4609.3164062
2: -1897.9423828, 2585.7343750, -1614.0108643, 2215.9260254, -4113.1997070, 4199.7451172
3: -2531.2509766, 3281.2028809, -2160.0268555, 2805.6811523, -5336.9321289, 5441.2294922
4: -2432.9736328, 3523.6503906, -2070.8305664, 3021.9294434, -5454.9033203, 5594.4804688

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5651475, upper bound: 4974.5591274
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5699652, upper bound: 4974.5651418
time: 1.17 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2736.8164062, 3220.3635254, -3018.6284180, 3584.5200195, -6317.0834961, 6236.5834961
1: -2293.7900391, 2579.5222168, -2537.3769531, 2871.2006836, -5164.4331055, 5116.8994141
2: -1867.3288574, 2542.7189941, -2066.4694824, 2834.5585938, -4700.5371094, 4608.3466797
3: -2489.8310547, 3226.5766602, -2766.8041992, 3590.5034180, -6080.0791016, 5991.8916016
4: -2393.4985352, 3464.8085938, -2655.9975586, 3864.5627441, -6258.0610352, 6120.8061523

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5656514, upper bound: 4974.5609406
time: 1.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5699652, upper bound: 4974.5655516
time: 1.19 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2764.6745605, 3255.9152832, -2357.7236328, 2805.4020996, -5570.0766602, 5613.6386719
1: -2318.4167480, 2607.5546875, -1988.6202393, 2245.2595215, -4563.6762695, 4596.1748047
2: -1886.9732666, 2571.3205566, -1615.2939453, 2219.2971191, -4106.2705078, 4186.6142578
3: -2516.8330078, 3262.0544434, -2162.6574707, 2809.2612305, -5326.0942383, 5424.7114258
4: -2418.8767090, 3504.6066895, -2072.8493652, 3027.2246094, -5446.1015625, 5577.4550781

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5750015, upper bound: 4974.5719961
time: 1.49 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5795403, upper bound: 4974.5757370
time: 1.26 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2747.6994629, 3234.2275391, -2569.2519531, 3052.8129883, -5800.5122070, 5803.4794922
1: -2303.6777344, 2590.2053223, -2167.3015137, 2444.4665527, -4748.1440430, 4757.5043945
2: -1874.9913330, 2553.9445801, -1760.7047119, 2416.5332031, -4291.3325195, 4314.6479492
3: -2500.5922852, 3240.2380371, -2356.9926758, 3057.2668457, -5557.8593750, 5597.2304688
4: -2403.2697754, 3480.6347656, -2259.6408691, 3296.4838867, -5699.7539062, 5740.2749023

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5535234, upper bound: 4974.5496454
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5664413, upper bound: 4974.5689750
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2824.3183594, 3326.6481934, -2409.7458496, 2863.7575684, -5687.6630859, 5736.3940430
1: -2367.9575195, 2663.7978516, -2030.1662598, 2290.7041016, -4658.6616211, 4693.9638672
2: -1927.3682861, 2626.1423340, -1649.7253418, 2263.7263184, -4190.5517578, 4275.8666992
3: -2570.7062988, 3332.9855957, -2206.7687988, 2866.8000488, -5437.5043945, 5539.7543945
4: -2470.5783691, 3578.2941895, -2115.8776855, 3086.2604980, -5556.8374023, 5694.1704102

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5712920, upper bound: 4974.5651269
time: 1.28 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5722260, upper bound: 4974.5667168
time: 1.25 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2781.5051270, 3273.6923828, -3050.7607422, 3621.6987305, -6398.9301758, 6322.1782227
1: -2331.2590332, 2621.6628418, -2564.6108398, 2900.6064453, -5231.3027344, 5186.2734375
2: -1897.6965332, 2584.2573242, -2088.5063477, 2863.5427246, -4759.8398438, 4671.9555664
3: -2530.5241699, 3279.8662109, -2795.1276855, 3627.4621582, -6157.4848633, 6073.3730469
4: -2432.2744141, 3520.9895020, -2683.1479492, 3903.2604980, -6335.5351562, 6204.1372070

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5702983, upper bound: 4974.5663638
time: 1.35 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5721251, upper bound: 4974.5668336
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2813.2832031, 3314.8852539, -2412.1074219, 2869.0119629, -5682.2944336, 5726.9926758
1: -2359.3540039, 2654.2238770, -2033.6940918, 2295.1191406, -4654.4721680, 4687.9174805
2: -1920.1488037, 2617.4311523, -1652.2159424, 2268.8710938, -4189.0190430, 4269.6464844
3: -2561.1496582, 3320.9694824, -2210.1667480, 2872.1643066, -5433.3134766, 5531.1357422
4: -2461.2199707, 3566.8044434, -2119.0139160, 3093.3405762, -5554.5605469, 5685.8173828

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5799627, upper bound: 4974.5801121
time: 1.43 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5800592, upper bound: 4974.5800592
time: 1.26 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2774.8039551, 3267.1972656, -3045.3215332, 3615.2412109, -6388.3637695, 6312.3603516
1: -2326.2802734, 2616.2871094, -2560.8144531, 2895.7185059, -5221.9990234, 5177.1015625
2: -1893.4282227, 2579.6650391, -2085.1743164, 2858.8051758, -4752.0083008, 4664.8393555
3: -2524.9719238, 3273.1127930, -2790.3366699, 3621.4758301, -6146.4477539, 6063.4467773
4: -2426.7509766, 3515.1394043, -2678.6860352, 3896.8374023, -6323.5874023, 6193.8251953

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5733238, upper bound: 4974.5678928
time: 1.11 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5796037, upper bound: 4974.5796037
time: 1.10 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.18 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5766162, upper bound: 4974.5790855
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5604101, upper bound: 4974.5572121
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5766162, upper bound: 4974.5790855
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5604101, upper bound: 4974.5572121
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5765641, upper bound: 4974.5784576
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5763282, upper bound: 4974.5773061
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5765641, upper bound: 4974.5784576
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5763282, upper bound: 4974.5773061
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5613301, upper bound: 4974.5697323
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5613301, upper bound: 4974.5697323
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5611489, upper bound: 4974.5693315
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5611048, upper bound: 4974.5676510
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5611489, upper bound: 4974.5693315
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5611048, upper bound: 4974.5676510
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5651475, upper bound: 4974.5591274
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5699652, upper bound: 4974.5651418
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5656514, upper bound: 4974.5609406
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5699652, upper bound: 4974.5655516
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5750015, upper bound: 4974.5719961
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5795403, upper bound: 4974.5757370
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5535234, upper bound: 4974.5496454
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5664413, upper bound: 4974.5689750
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5712920, upper bound: 4974.5651269
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5722260, upper bound: 4974.5667168
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5702983, upper bound: 4974.5663638
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5721251, upper bound: 4974.5668336
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5799627, upper bound: 4974.5801121
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5800592, upper bound: 4974.5800592
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5733238, upper bound: 4974.5678928
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -4974.5796037, upper bound: 4974.5796037

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2365.4487305, 2812.8911133, -2391.6674805, 2843.4316406, -5208.8793945, 5204.5576172
1: -1993.4818115, 2251.1489258, -2014.8992920, 2275.4750977, -4268.9570312, 4266.0468750
2: -1619.9233398, 2224.6743164, -1637.5616455, 2248.6181641, -3868.5410156, 3862.2353516
3: -2168.3220215, 2816.0461426, -2191.8432617, 2846.3413086, -5014.6625977, 5007.8896484
4: -2078.6369629, 3033.1833496, -2101.3239746, 3065.7294922, -5144.3662109, 5134.5068359

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5837071, upper bound: 4974.5795207
time: 1.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5818547, upper bound: 4974.5794396
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2378.6835938, 2826.4248047, -2365.6591797, 2812.1210938, -5190.8046875, 5192.0830078
1: -2005.2462158, 2262.6330566, -1993.6776123, 2250.6164551, -4255.8627930, 4256.3105469
2: -1628.8576660, 2235.5847168, -1619.8763428, 2224.0063477, -3852.8640137, 3855.4609375
3: -2179.9965820, 2830.5126953, -2168.3510742, 2815.2861328, -4995.2827148, 4998.8637695
4: -2089.7336426, 3048.1838379, -2078.5129395, 3032.5949707, -5122.3286133, 5126.6958008

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5868193, upper bound: 4974.5840910
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5852342, upper bound: 4974.5839624
time: 1.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2365.4487305, 2812.8911133, -2750.8146973, 3236.5961914, -5602.0444336, 5563.1201172
1: -1993.4818115, 2251.1489258, -2304.8317871, 2592.3591309, -4585.8408203, 4555.9804688
2: -1619.9233398, 2224.6743164, -1876.8177490, 2555.3234863, -4175.2470703, 4100.7294922
3: -2168.3220215, 2816.0461426, -2502.6325684, 3242.3286133, -5410.6499023, 5318.6787109
4: -2078.6369629, 3033.1833496, -2405.7216797, 3481.5144043, -5560.1513672, 5438.9052734

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5760160, upper bound: 4974.5757063
time: 1.24 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5747589, upper bound: 4974.5753021
time: 3.18 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2378.6835938, 2826.4248047, -2701.8942871, 3177.1430664, -5555.8266602, 5528.3193359
1: -2005.2462158, 2262.6330566, -2263.8620605, 2544.9035645, -4550.1499023, 4526.4951172
2: -1628.8576660, 2235.5847168, -1843.2742920, 2508.2790527, -4137.1367188, 4078.8588867
3: -2179.9965820, 2830.5126953, -2457.5058594, 3183.1801758, -5363.1767578, 5288.0185547
4: -2089.7336426, 3048.1838379, -2362.6530762, 3417.4799805, -5507.2128906, 5410.8349609

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5539868, upper bound: 4974.5552404
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5603753, upper bound: 4974.5570735
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2440.0622559, 2899.9201660, -2453.6955566, 2916.2587891, -5356.3212891, 5353.6157227
1: -2055.5908203, 2319.7543945, -2066.3334961, 2333.0388184, -4388.6298828, 4386.0878906
2: -1670.5916748, 2292.6508789, -1679.5523682, 2305.4055176, -3975.9965820, 3972.2028809
3: -2234.6772461, 2902.6201172, -2247.3874512, 2919.1225586, -5153.7998047, 5150.0078125
4: -2142.7946777, 3125.3254395, -2154.9213867, 3142.7282715, -5285.5229492, 5280.2460938

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5855839, upper bound: 4974.5848452
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5848814, upper bound: 4974.5848452
time: 1.43 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3128.4650879, 3715.1127930, -2435.9941406, 2894.4062500, -6022.8710938, 6151.1069336
1: -2630.1567383, 2975.4355469, -2051.3662109, 2315.5612793, -4945.7177734, 5026.8012695
2: -2142.0791016, 2937.6376953, -1667.3400879, 2288.0651855, -4430.1445312, 4604.9775391
3: -2867.0244141, 3720.9094238, -2230.7255859, 2897.3332520, -5764.3574219, 5951.6337891
4: -2752.2924805, 4004.2709961, -2139.1184082, 3118.9770508, -5871.2685547, 6143.3891602

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5854860, upper bound: 4974.5848351
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5848756, upper bound: 4974.5848351
time: 1.28 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2440.0622559, 2899.9201660, -2806.2023926, 3302.6496582, -5742.7119141, 5706.1225586
1: -2055.5908203, 2319.7543945, -2351.3073730, 2644.8212891, -4700.4121094, 4671.0610352
2: -1670.5916748, 2292.6508789, -1914.5151367, 2607.0205078, -4277.6123047, 4207.1660156
3: -2234.6772461, 2902.6201172, -2553.1232910, 3308.4912109, -5543.1684570, 5455.7431641
4: -2142.7946777, 3125.3254395, -2454.1743164, 3551.7243652, -5694.5185547, 5579.4995117

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5764271, upper bound: 4974.5771536
time: 1.37 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5751142, upper bound: 4974.5765306
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3128.4650879, 3715.1127930, -2764.4758301, 3251.2128906, -6378.6572266, 6476.4936523
1: -2630.1567383, 2975.4355469, -2315.6386719, 2603.8872070, -5234.0439453, 5291.0737305
2: -2142.0791016, 2937.6376953, -1885.6080322, 2566.3344727, -4707.9707031, 4822.4057617
3: -2867.0244141, 3720.9094238, -2514.0639648, 3256.8864746, -6123.0341797, 6234.9721680
4: -2752.2924805, 4004.2709961, -2416.9121094, 3496.0522461, -6248.3437500, 6421.1831055

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5762970, upper bound: 4974.5768999
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5750055, upper bound: 4974.5762988
time: 1.33 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2461.8269043, 2926.8020020, -2637.0578613, 3139.8874512, -5601.7143555, 5563.8583984
1: -2073.5930176, 2341.1269531, -2223.1252441, 2511.8015137, -4585.3945312, 4564.2514648
2: -1685.2219238, 2313.9665527, -1806.2260742, 2483.8066406, -4169.0283203, 4120.1923828
3: -2255.2331543, 2929.0500488, -2418.6188965, 3143.0139160, -5398.2470703, 5347.6689453
4: -2162.1079102, 3154.4299316, -2318.0952148, 3387.1264648, -5549.2329102, 5472.5253906

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5827046, upper bound: 4974.5820680
time: 1.26 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5836033, upper bound: 4974.5856826
time: 1.33 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5831766, upper bound: 4974.5856824
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3155.8132324, 3749.9401855, -2616.1171875, 3113.9653320, -6269.7783203, 6366.0576172
1: -2652.8469238, 3002.8391113, -2205.3537598, 2491.0949707, -5143.9414062, 5208.1928711
2: -2160.6340332, 2965.2431641, -1791.7325439, 2463.2658691, -4623.7626953, 4756.9755859
3: -2893.2514648, 3754.6979980, -2398.8330078, 3117.1289062, -6010.3803711, 6153.5302734
4: -2777.0266113, 4042.4172363, -2299.3413086, 3358.9877930, -6136.0146484, 6341.7563477

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5832203, upper bound: 4974.5821172
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5835408, upper bound: 4974.5837133
time: 1.18 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5830683, upper bound: 4974.5836999
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2461.8269043, 2926.8020020, -2985.4533691, 3520.0585938, -5981.8857422, 5912.2553711
1: -2073.5930176, 2341.1269531, -2504.9968262, 2818.0756836, -4891.6689453, 4846.1230469
2: -1685.2219238, 2313.9665527, -2038.2532959, 2780.1708984, -4465.3920898, 4351.8125000
3: -2255.2331543, 2929.0500488, -2719.5947266, 3526.0798340, -5781.3129883, 5648.6445312
4: -2162.1079102, 3154.4299316, -2613.1486816, 3789.1462402, -5951.2539062, 5767.5786133

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5578980, upper bound: 4974.5646379
time: 1.40 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
time: 1.29 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
time: 1.43 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3155.8132324, 3749.9401855, -2942.8041992, 3467.5690918, -6621.0639648, 6689.1171875
1: -2652.8469238, 3002.8391113, -2468.4328613, 2776.3125000, -5429.1591797, 5470.9921875
2: -2160.6340332, 2965.2431641, -2008.7359619, 2738.6784668, -4898.4067383, 4972.7866211
3: -2893.2514648, 3754.6979980, -2679.6794434, 3473.4477539, -6365.6430664, 6434.3774414
4: -2777.0266113, 4042.4172363, -2575.1420898, 3732.3947754, -6509.4213867, 6617.4282227

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5578840, upper bound: 4974.5644994
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
time: 1.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
time: 1.12 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2464.5917969, 2927.6984863, -2628.1870117, 3133.1035156, -5597.6953125, 5555.8857422
1: -2076.7194824, 2342.6833496, -2217.0593262, 2506.5134277, -4583.2329102, 4559.7426758
2: -1687.0532227, 2315.1730957, -1801.1420898, 2479.2058105, -4166.2587891, 4116.3144531
3: -2257.1408691, 2930.8322754, -2412.8452148, 3136.1076660, -5393.2480469, 5343.6777344
4: -2163.8740234, 3156.1679688, -2311.9606934, 3381.3171387, -5545.1914062, 5468.1289062

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5834444, upper bound: 4974.5845305
time: 1.22 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5829062, upper bound: 4974.5845800
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3160.4741211, 3751.5139160, -2612.8713379, 3113.7358398, -6274.2084961, 6364.3833008
1: -2656.5449219, 3004.8278809, -2203.9179688, 2491.0776367, -5147.6215820, 5208.7456055
2: -2163.2370605, 2966.4489746, -1790.4249268, 2463.7800293, -4627.0170898, 4756.8735352
3: -2895.6416016, 3757.3364258, -2398.1154785, 3116.7863770, -6012.4277344, 6155.4521484
4: -2779.7556152, 4043.6340332, -2298.0976562, 3360.0937500, -6139.8491211, 6341.7309570

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5834444, upper bound: 4974.5829463
time: 1.28 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5829062, upper bound: 4974.5829062
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2464.5917969, 2927.6984863, -2962.7919922, 3495.4704590, -5960.0625000, 5890.4902344
1: -2076.7194824, 2342.6833496, -2486.2263184, 2798.2309570, -4874.9497070, 4828.9091797
2: -1687.0532227, 2315.1730957, -2023.2023926, 2761.1105957, -4448.1635742, 4338.3740234
3: -2257.1408691, 2930.8322754, -2700.1181641, 3501.4550781, -5758.5957031, 5630.9501953
4: -2163.8740234, 3156.1679688, -2594.2658691, 3763.4296875, -5927.3037109, 5750.4331055

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5578183, upper bound: 4974.5644913
time: 1.14 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611048, upper bound: 4974.5676510
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611048, upper bound: 4974.5676510
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3160.4741211, 3751.5139160, -2922.2399902, 3445.8581543, -6606.3305664, 6672.6362305
1: -2656.5449219, 3004.8278809, -2451.5676270, 2758.7163086, -5415.2612305, 5456.3955078
2: -2163.2370605, 2966.4489746, -1995.1854248, 2721.8234863, -4885.0605469, 4961.4501953
3: -2895.6416016, 3757.3364258, -2662.2685547, 3451.6550293, -6346.9091797, 6419.6049805
4: -2779.7556152, 4043.6340332, -2558.2426758, 3709.7031250, -6489.4584961, 6601.8745117

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5578183, upper bound: 4974.5644641
time: 1.50 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611048, upper bound: 4974.5676510
time: 1.27 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611048, upper bound: 4974.5676510
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2745.3583984, 3231.2412109, -2325.9602051, 2764.8740234, -5509.8774414, 5557.2011719
1: -2301.3891602, 2587.9724121, -1960.3609619, 2212.6987305, -4514.0874023, 4548.3334961
2: -1873.3717041, 2551.1723633, -1592.9416504, 2186.4904785, -4059.1950684, 4144.1137695
3: -2498.0041504, 3237.6176758, -2131.6025391, 2768.4990234, -5266.5029297, 5369.2202148
4: -2401.1691895, 3476.4362793, -2043.6213379, 2981.6818848, -5382.8510742, 5520.0561523

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5578967, upper bound: 4974.5519009
time: 1.21 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5578967, upper bound: 4974.5581355
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2734.4736328, 3212.1718750, -2309.5852051, 2744.5224609, -5477.2319336, 5521.7568359
1: -2290.8786621, 2573.6379395, -1946.6323242, 2196.6176758, -4487.4960938, 4520.2705078
2: -1865.2515869, 2535.7790527, -1581.6729736, 2170.4963379, -4034.2956543, 4117.4516602
3: -2485.3962402, 3219.1635742, -2116.2478027, 2748.3508301, -5233.7460938, 5334.6967773
4: -2390.0007324, 3454.7084961, -2029.0261230, 2960.0395508, -5349.9433594, 5483.7333984

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5680387, upper bound: 4974.5640394
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5680387, upper bound: 4974.5648689
time: 1.25 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2699.2653809, 3174.5688477, -2998.8359375, 3560.4645996, -6255.3862305, 6170.9179688
1: -2261.9223633, 2542.8876953, -2520.5087891, 2852.0539551, -5113.3583984, 5063.3959961
2: -1841.4970703, 2506.3842773, -2052.8532715, 2815.5402832, -4655.6689453, 4558.3540039
3: -2454.8894043, 3180.7272949, -2748.4426270, 3566.4379883, -6021.0415039, 5927.6582031
4: -2360.0773926, 3415.1669922, -2638.4499512, 3838.5627441, -6198.6386719, 6053.6166992

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611718, upper bound: 4974.5553791
time: 1.39 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5655200, upper bound: 4974.5609105
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2691.8068848, 3160.0275879, -2958.2116699, 3513.1127930, -6199.1494141, 6114.8686523
1: -2254.4223633, 2532.1000977, -2486.5217285, 2814.0349121, -5066.9741211, 5018.3007812
2: -1835.7934570, 2494.5480957, -2025.3498535, 2777.9719238, -4611.5981445, 4518.6186523
3: -2445.5646973, 3166.8347168, -2711.4907227, 3519.2285156, -5964.1269531, 5875.7397461
4: -2352.0847168, 3398.2900391, -2603.0668945, 3787.2509766, -6138.4423828, 6001.3564453

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5651522, upper bound: 4974.5626108
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5651522, upper bound: 4974.5649731
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2737.9902344, 3223.7224121, -2349.9316406, 2795.8569336, -5533.8471680, 5573.6542969
1: -2295.8488770, 2581.8300781, -1981.9763184, 2237.6391602, -4533.4882812, 4563.8066406
2: -1868.6403809, 2545.8610840, -1609.9138184, 2211.7392578, -4080.3796387, 4155.7749023
3: -2492.0668945, 3229.8518066, -2155.3576660, 2799.6884766, -5291.7548828, 5385.2089844
4: -2395.2299805, 3469.7724609, -2065.8967285, 3016.8781738, -5412.1069336, 5535.6689453

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5742767, upper bound: 4974.5716875
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5742767, upper bound: 4974.5716875
time: 1.27 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3346.7038574, 3948.0656738, -2326.3937988, 2766.8815918, -6110.9570312, 6274.4594727
1: -2803.4687500, 3164.3496094, -1962.0640869, 2214.5368652, -5018.0048828, 5126.4130859
2: -2285.8647461, 3119.1013184, -1593.7120361, 2188.7543945, -4472.2749023, 4712.8134766
3: -3052.9592285, 3955.5375977, -2133.2690430, 2770.8073730, -5823.7666016, 6088.3686523
4: -2935.6335449, 4250.9760742, -2044.9578857, 2985.4152832, -5918.8955078, 6295.9335938

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5795403, upper bound: 4974.5757370
time: 1.13 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5795403, upper bound: 4974.5757370
time: 1.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2698.5217285, 3175.5578613, -2549.1323242, 3028.7646484, -5727.2861328, 5724.6904297
1: -2262.3894043, 2543.2048340, -2150.3300781, 2425.2121582, -4687.6010742, 4693.5346680
2: -1841.2922363, 2507.5698242, -1746.9096680, 2397.5429688, -4238.5625000, 4254.4790039
3: -2455.4421387, 3181.3178711, -2338.5068359, 3033.1430664, -5488.5849609, 5519.8247070
4: -2359.8217773, 3417.4494629, -2241.8928223, 3270.5766602, -5630.3979492, 5659.3422852

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5516077, upper bound: 4974.5485703
time: 4.32 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5516077, upper bound: 4974.5485703
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2748.9633789, 3239.4362793, -2494.6145020, 2964.9619141, -5713.9238281, 5734.0507812
1: -2305.8415527, 2594.3107910, -2104.6599121, 2373.8630371, -4679.7041016, 4698.9702148
2: -1876.7130127, 2558.8110352, -1709.6685791, 2346.9418945, -4223.4028320, 4268.4794922
3: -2503.8378906, 3244.3737793, -2288.9748535, 2968.6694336, -5472.5068359, 5533.3481445
4: -2405.7543945, 3488.3969727, -2194.2045898, 3201.8347168, -5607.5888672, 5682.6010742

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5664413, upper bound: 4974.5689750
time: 1.33 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5664413, upper bound: 4974.5689750
time: 1.41 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2789.5236816, 3284.1374512, -2378.7121582, 2826.2333984, -5615.3256836, 5662.8496094
1: -2338.4353027, 2629.7915039, -2003.8333740, 2260.6601562, -4599.0957031, 4633.6245117
2: -1903.4191895, 2592.4099121, -1628.4196777, 2233.9577637, -4136.8369141, 4220.8295898
3: -2538.3073730, 3290.4353027, -2178.0546875, 2829.2470703, -5367.5541992, 5468.4887695
4: -2439.6191406, 3532.2414551, -2088.3388672, 3045.5878906, -5485.2060547, 5620.5800781

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5712920, upper bound: 4974.5651269
time: 1.11 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5712921, upper bound: 4974.5651086
time: 1.32 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2775.5876465, 3262.6887207, -2359.7941895, 2803.7739258, -5577.4711914, 5622.4829102
1: -2325.7036133, 2613.4711914, -1987.9691162, 2242.8530273, -4568.5561523, 4601.4404297
2: -1893.5052490, 2575.2241211, -1615.5228271, 2216.3400879, -4108.5268555, 4190.7456055
3: -2523.5810547, 3269.5661621, -2160.6870117, 2806.8955078, -5330.4765625, 5429.4487305
4: -2426.1928711, 3508.0554199, -2071.8322754, 3021.6418457, -5447.7919922, 5579.8876953

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5722260, upper bound: 4974.5667168
time: 1.12 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5717715, upper bound: 4974.5667126
time: 1.36 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2735.7165527, 3216.8386230, -3033.0563965, 3599.3061523, -6330.6455078, 6247.5454102
1: -2292.1843262, 2576.4140625, -2549.4626465, 2882.9130859, -5174.4472656, 5125.8764648
2: -1866.0583496, 2539.3098145, -2076.2141113, 2845.9118652, -4710.5078125, 4614.6289062
3: -2487.8032227, 3222.9799805, -2778.4951172, 3605.0659180, -6092.3354492, 5999.8071289
4: -2391.4851074, 3459.7612305, -2667.2358398, 3879.2878418, -6270.7646484, 6126.9970703

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5693289, upper bound: 4974.5663411
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5693289, upper bound: 4974.5663638
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2915.5732422, 3431.7204590, -3006.4370117, 3567.1887207, -6478.5454102, 6436.0366211
1: -2445.5087891, 2748.5781250, -2526.7387695, 2857.0439453, -5302.1123047, 5275.3168945
2: -1990.3540039, 2711.1062012, -2057.6921387, 2820.3139648, -4809.3422852, 4767.8686523
3: -2654.9719238, 3437.7229004, -2753.7368164, 3572.8286133, -6227.3935547, 6189.8139648
4: -2551.6250000, 3695.0751953, -2643.4565430, 3844.2844238, -6395.9091797, 6338.5307617

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5721251, upper bound: 4974.5668336
time: 1.21 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5721251, upper bound: 4974.5668336
time: 1.52 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2717.6398926, 3206.0737305, -2388.5339355, 2840.1953125, -5557.8349609, 5594.6074219
1: -2281.2268066, 2567.3503418, -2013.5163574, 2271.8764648, -4553.1035156, 4580.8666992
2: -1855.6198730, 2532.5212402, -1635.7775879, 2245.8745117, -4101.4941406, 4168.2983398
3: -2476.9025879, 3212.7102051, -2188.0390625, 2843.2106934, -5320.1127930, 5400.7490234
4: -2378.9072266, 3452.4316406, -2097.8632812, 3061.8977051, -5440.8046875, 5550.2949219

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5780645, upper bound: 4974.5746879
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5649770, upper bound: 4974.5613005
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2755.1110840, 3245.7258301, -2384.1696777, 2835.2639160, -5590.3750000, 5629.8955078
1: -2309.8959961, 2598.6428223, -2009.5382080, 2267.9731445, -4577.8691406, 4608.1811523
2: -1880.0849609, 2562.6564941, -1632.7247314, 2242.0485840, -4122.0913086, 4195.3798828
3: -2507.7846680, 3251.4033203, -2184.0898438, 2838.1452637, -5345.9296875, 5435.4931641
4: -2410.0434570, 3492.0270996, -2094.0695801, 3056.6879883, -5466.7309570, 5586.0957031

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5792898, upper bound: 4974.5757391
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5723145, upper bound: 4974.5743537
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2731.0734863, 3214.2810059, -3024.7834473, 3590.2243652, -6319.5322266, 6238.8339844
1: -2289.2700195, 2573.9868164, -2543.3134766, 2875.7875977, -5165.0576172, 5117.2988281
2: -1863.4250488, 2537.7497559, -2071.0268555, 2839.0065918, -4702.1953125, 4608.7763672
3: -2484.4765625, 3220.1674805, -2771.2568359, 3596.4323730, -6080.9086914, 5991.3964844
4: -2387.9948730, 3457.8808594, -2660.4426270, 3869.8127441, -6257.8076172, 6118.3232422

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5733211, upper bound: 4974.5678928
time: 1.50 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5733211, upper bound: 4974.5678928
time: 1.52 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2728.6113281, 3206.3251953, -2990.2211914, 3550.2563477, -6275.7343750, 6195.4633789
1: -2286.3229980, 2568.3803711, -2514.4677734, 2843.6501465, -5129.9716797, 5082.8471680
2: -1861.4311523, 2531.2641602, -2047.7015381, 2807.2922363, -4667.7060547, 4578.6948242
3: -2480.3161621, 3212.7707520, -2739.9667969, 3556.5615234, -6036.8769531, 5951.6821289
4: -2384.8671875, 3448.5402832, -2630.4401855, 3826.4641113, -6211.3310547, 6078.9804688

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5796037, upper bound: 4974.5796037
time: 1.32 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5796037, upper bound: 4974.5796037
time: 1.02 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.42 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5837071, upper bound: 4974.5795207
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5818547, upper bound: 4974.5794396
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5868193, upper bound: 4974.5840910
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5852342, upper bound: 4974.5839624
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5760160, upper bound: 4974.5757063
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5747589, upper bound: 4974.5753021
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5539868, upper bound: 4974.5552404
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5603753, upper bound: 4974.5570735
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5855839, upper bound: 4974.5848452
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5848814, upper bound: 4974.5848452
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5854860, upper bound: 4974.5848351
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5848756, upper bound: 4974.5848351
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5764271, upper bound: 4974.5771536
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5751142, upper bound: 4974.5765306
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5762970, upper bound: 4974.5768999
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5750055, upper bound: 4974.5762988
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5836033, upper bound: 4974.5856826
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5831766, upper bound: 4974.5856824
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5835408, upper bound: 4974.5837133
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5830683, upper bound: 4974.5836999
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5834444, upper bound: 4974.5845305
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5829062, upper bound: 4974.5845800
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5834444, upper bound: 4974.5829463
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5829062, upper bound: 4974.5829062
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5611048, upper bound: 4974.5676510
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5611048, upper bound: 4974.5676510
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5611048, upper bound: 4974.5676510
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5611048, upper bound: 4974.5676510
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5578967, upper bound: 4974.5519009
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5578967, upper bound: 4974.5581355
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5680387, upper bound: 4974.5640394
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5680387, upper bound: 4974.5648689
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5611718, upper bound: 4974.5553791
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5655200, upper bound: 4974.5609105
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5651522, upper bound: 4974.5626108
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5651522, upper bound: 4974.5649731
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5742767, upper bound: 4974.5716875
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5742767, upper bound: 4974.5716875
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5795403, upper bound: 4974.5757370
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5795403, upper bound: 4974.5757370
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5516077, upper bound: 4974.5485703
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5516077, upper bound: 4974.5485703
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5664413, upper bound: 4974.5689750
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5664413, upper bound: 4974.5689750
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5712920, upper bound: 4974.5651269
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5712921, upper bound: 4974.5651086
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5722260, upper bound: 4974.5667168
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5717715, upper bound: 4974.5667126
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5693289, upper bound: 4974.5663411
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5693289, upper bound: 4974.5663638
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5721251, upper bound: 4974.5668336
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5721251, upper bound: 4974.5668336
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5780645, upper bound: 4974.5746879
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5649770, upper bound: 4974.5613005
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5792898, upper bound: 4974.5757391
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5723145, upper bound: 4974.5743537
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5733211, upper bound: 4974.5678928
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5733211, upper bound: 4974.5678928
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5796037, upper bound: 4974.5796037
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -4974.5796037, upper bound: 4974.5796037

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2342.4160156, 2784.4406738, -2346.1997070, 2787.1247559, -5129.5410156, 5130.6391602
1: -1973.9979248, 2228.5058594, -1976.3133545, 2230.6577148, -4204.6557617, 4204.8193359
2: -1604.0263672, 2202.1604004, -1606.1118164, 2204.0375977, -3808.0639648, 3808.2722168
3: -2146.9370117, 2787.7346191, -2149.4877930, 2790.2463379, -4937.1831055, 4937.2226562
4: -2058.1782227, 3002.5708008, -2060.8198242, 3005.0712891, -5063.2490234, 5063.3906250

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5831660, upper bound: 4974.5788901
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5818715, upper bound: 4974.5773618
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2330.9453125, 2770.1950684, -2558.4038086, 3038.3908691, -5369.3359375, 5328.5986328
1: -1963.8356934, 2217.2250977, -2156.9692383, 2432.8391113, -4396.6748047, 4374.1938477
2: -1596.0476074, 2191.0373535, -1752.5950928, 2404.9079590, -4000.9553223, 3943.6323242
3: -2136.0168457, 2773.2634277, -2345.7680664, 3042.2658691, -5178.2822266, 5119.0312500
4: -2047.8085938, 2987.1516113, -2249.0275879, 3279.6894531, -5327.4975586, 5236.1787109

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5816186, upper bound: 4974.5788861
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5804739, upper bound: 4974.5773596
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2370.0935059, 2815.8820801, -2328.1374512, 2766.1962891, -5136.2900391, 5144.0195312
1: -1997.9284668, 2254.2145996, -1961.7169189, 2213.9025879, -4211.8300781, 4215.9316406
2: -1622.9193115, 2227.2395020, -1593.9366455, 2187.6252441, -3810.5444336, 3821.1760254
3: -2171.9382324, 2819.9453125, -2133.1953125, 2769.2290039, -4941.1669922, 4953.1401367
4: -2082.0546875, 3036.7634277, -2044.9832764, 2982.8112793, -5064.8652344, 5081.7460938

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5848655, upper bound: 4974.5834058
time: 1.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5853422, upper bound: 4974.5833281
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2351.6269531, 2793.1081543, -3035.7119141, 3603.2832031, -5954.9101562, 5828.8183594
1: -1982.2458496, 2236.0708008, -2550.8979492, 2886.1855469, -4868.4316406, 4786.9687500
2: -1610.1887207, 2209.1877441, -2077.7514648, 2849.1159668, -4459.3037109, 4286.9389648
3: -2154.4985352, 2797.2470703, -2781.6098633, 3609.0622559, -5763.5605469, 5578.8569336
4: -2065.5974121, 3012.0087891, -2670.5288086, 3883.9680176, -5949.5654297, 5682.5371094

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5841188, upper bound: 4974.5832350
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5839181, upper bound: 4974.5828022
time: 1.40 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2342.4160156, 2784.4406738, -2710.0273438, 3186.1511230, -5528.5673828, 5493.8525391
1: -1973.9979248, 2228.5058594, -2270.1918945, 2552.1486816, -4526.1464844, 4498.6977539
2: -1604.0263672, 2202.1604004, -1848.6131592, 2515.4304199, -4119.4555664, 4049.9858398
3: -2146.9370117, 2787.7346191, -2464.7382812, 3191.8461914, -5338.7822266, 5252.4726562
4: -2058.1782227, 3002.5708008, -2369.3554688, 3427.2502441, -5485.4282227, 5371.9262695

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5760160, upper bound: 4974.5757063
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5760160, upper bound: 4974.5757063
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2330.9453125, 2770.1950684, -2892.2521973, 3404.3168945, -5735.2622070, 5662.0019531
1: -1963.8356934, 2217.2250977, -2425.6730957, 2727.2241211, -4691.0590820, 4642.8984375
2: -1596.0476074, 2191.0373535, -1974.5928955, 2689.8793945, -4285.9267578, 4164.9174805
3: -2136.0168457, 2773.2634277, -2634.2529297, 3409.9680176, -5545.9838867, 5407.5161133
4: -2047.8085938, 2987.1516113, -2531.9116211, 3666.1740723, -5713.9824219, 5519.0625000

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5747589, upper bound: 4974.5753021
time: 1.23 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5747589, upper bound: 4974.5753021
time: 3.13 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2340.0522461, 2779.1296387, -2642.3564453, 3103.5332031, -5443.5849609, 5421.4858398
1: -1972.4688721, 2224.7451172, -2213.2453613, 2485.8100586, -4458.2773438, 4437.9902344
2: -1602.0656738, 2197.9714355, -1801.9095459, 2449.5954590, -4051.6611328, 3999.8808594
3: -2143.8378906, 2783.1721191, -2401.4550781, 3109.3461914, -5253.1840820, 5184.6269531
4: -2055.1350098, 2996.9028320, -2309.1030273, 3337.5109863, -5392.6459961, 5306.0058594

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5539868, upper bound: 4974.5552404
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5539868, upper bound: 4974.5552404
time: 1.37 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2347.1909180, 2789.3159180, -2673.0397949, 3143.5781250, -5490.7690430, 5462.3554688
1: -1978.8616943, 2232.8784180, -2239.4226074, 2517.9272461, -4496.7890625, 4472.3007812
2: -1607.3557129, 2206.2834473, -1823.4205322, 2481.6032715, -4088.9589844, 4029.7033691
3: -2151.0996094, 2793.3137207, -2430.9953613, 3149.4418945, -5300.5405273, 5224.3090820
4: -2062.1035156, 3008.1223145, -2337.2211914, 3380.8493652, -5442.9531250, 5345.3437500

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5603753, upper bound: 4974.5570735
time: 1.42 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5603753, upper bound: 4974.5570735
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2416.7790527, 2870.6235352, -2406.5900879, 2857.3662109, -5274.1455078, 5277.2128906
1: -2035.7236328, 2296.4377441, -2026.2399902, 2286.1850586, -4321.9082031, 4322.6777344
2: -1654.4144287, 2269.4291992, -1646.9080811, 2258.7976074, -3913.2119141, 3916.3371582
3: -2212.7856445, 2873.3005371, -2203.2714844, 2860.2438965, -5073.0288086, 5076.5722656
4: -2121.8601074, 3093.6804199, -2112.7165527, 3079.2160645, -5201.0761719, 5206.3964844

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5845010, upper bound: 4974.5840226
time: 1.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5845283, upper bound: 4974.5832212
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2410.9951172, 2864.3569336, -2609.9040527, 3100.1811523, -5511.1762695, 5474.2607422
1: -2030.8752441, 2291.3989258, -2199.8012695, 2481.3908691, -4512.2661133, 4491.2001953
2: -1650.5212402, 2264.5368652, -1787.6547852, 2452.9836426, -4103.5043945, 4052.1914062
3: -2207.8906250, 2866.9846191, -2392.4228516, 3103.8591309, -5311.7485352, 5259.4067383
4: -2116.9343262, 3087.0571289, -2293.8972168, 3344.8452148, -5461.7792969, 5380.9541016

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5834351, upper bound: 4974.5839569
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5834687, upper bound: 4974.5832212
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3109.7060547, 3691.5964355, -2386.6025391, 2832.7687988, -5942.4746094, 6078.1992188
1: -2614.1220703, 2956.7895508, -2009.3339844, 2266.5058594, -4880.6279297, 4966.1230469
2: -2129.0761719, 2919.0524902, -1633.1270752, 2239.2822266, -4368.3583984, 4552.1791992
3: -2849.4645996, 3697.4089355, -2184.4799805, 2835.6838379, -5685.1484375, 5881.8886719
4: -2735.5112305, 3978.9826660, -2094.8823242, 3052.4892578, -5788.0004883, 6073.8647461

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5832465, upper bound: 4974.5820461
time: 1.52 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5814887, upper bound: 4974.5791373
time: 1.37 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3087.3059082, 3664.8557129, -2585.7500000, 3070.0253906, -6157.3310547, 6250.6054688
1: -2595.1188965, 2935.1877441, -2179.2211914, 2457.2297363, -5052.3471680, 5114.4091797
2: -2113.5495605, 2897.7932129, -1770.8035889, 2429.0178223, -4542.5673828, 4668.5966797
3: -2828.7941895, 3670.5053711, -2369.4084473, 3073.9191895, -5902.7128906, 6039.9140625
4: -2715.5827637, 3950.0422363, -2272.0947266, 3312.0410156, -6027.6230469, 6222.1367188

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5826237, upper bound: 4974.5820387
time: 1.35 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5812804, upper bound: 4974.5791373
time: 1.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2416.7790527, 2870.6235352, -2763.3269043, 3249.2971191, -5666.0747070, 5633.9492188
1: -2035.7236328, 2296.4377441, -2314.7519531, 2602.3137207, -4638.0366211, 4611.1889648
2: -1654.4144287, 2269.4291992, -1884.8181152, 2564.8388672, -4219.2529297, 4154.2470703
3: -2212.7856445, 2873.3005371, -2513.0266113, 3255.0263672, -5467.8120117, 5386.3271484
4: -2121.8601074, 3093.6804199, -2415.8234863, 3494.2717285, -5616.1318359, 5509.5034180

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5526478, upper bound: 4974.5501025
time: 1.53 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5734558, upper bound: 4974.5724649
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2410.9951172, 2864.3569336, -2942.1958008, 3463.3898926, -5874.3842773, 5806.5527344
1: -2030.8752441, 2291.3989258, -2467.3913574, 2773.9682617, -4804.8427734, 4758.7900391
2: -1650.5212402, 2264.5368652, -2008.5155029, 2736.0576172, -4386.5791016, 4273.0512695
3: -2207.8906250, 2866.9846191, -2679.5776367, 3469.1496582, -5677.0380859, 5546.5620117
4: -2116.9343262, 3087.0571289, -2575.2719727, 3728.8508301, -5845.7846680, 5662.3291016

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5494396, upper bound: 4974.5478813
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5723086, upper bound: 4974.5717135
time: 1.86 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3109.7060547, 3691.5964355, -2718.1562500, 3193.4216309, -6302.0678711, 6406.5820312
1: -2614.1220703, 2956.7895508, -2276.1118164, 2557.8557129, -5171.9775391, 5232.9008789
2: -2129.0761719, 2919.0524902, -1853.5117188, 2520.6110840, -4649.1918945, 4771.6713867
3: -2849.4645996, 3697.4089355, -2470.6972656, 3199.0036621, -6047.5571289, 6168.1059570
4: -2735.5112305, 3978.9826660, -2375.4458008, 3433.7648926, -6169.2749023, 6354.4287109

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5465131, upper bound: 4974.5443865
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5730622, upper bound: 4974.5725485
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3087.3059082, 3664.8557129, -2899.0212402, 3410.1079102, -6496.5073242, 6560.8481445
1: -2595.1188965, 2935.1877441, -2430.6391602, 2731.4865723, -5326.6054688, 5365.8271484
2: -2113.5495605, 2897.7932129, -1978.5906982, 2693.9165039, -4806.8476562, 4875.5664062
3: -2828.7941895, 3670.5053711, -2639.0454102, 3415.7009277, -6243.5786133, 6309.5502930
4: -2715.5827637, 3950.0422363, -2536.6018066, 3671.2353516, -6386.8168945, 6486.6430664

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5456820, upper bound: 4974.5443403
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5722124, upper bound: 4974.5718670
time: 1.41 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2446.2922363, 2908.3720703, -2616.2014160, 3114.9497070, -5561.2421875, 5524.5732422
1: -2060.5021973, 2326.4016113, -2205.4750977, 2491.8896484, -4552.3916016, 4531.8769531
2: -1674.6033936, 2299.4072266, -1791.9163818, 2464.1005859, -4138.7031250, 4091.3234863
3: -2241.0502930, 2910.5795898, -2399.4785156, 3118.0229492, -5359.0717773, 5310.0581055
4: -2148.5085449, 3134.5690918, -2299.7543945, 3360.2397461, -5508.7480469, 5434.3232422

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5800190, upper bound: 4974.5799912
time: 1.36 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5801444, upper bound: 4974.5797807
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2438.4565430, 2899.9814453, -2634.3889160, 3139.6520996, -5578.1083984, 5534.3701172
1: -2054.2443848, 2319.6945801, -2221.6044922, 2511.1704102, -4565.4150391, 4541.2988281
2: -1669.4293213, 2292.8730469, -1804.9753418, 2483.3400879, -4152.7695312, 4097.8481445
3: -2234.3562012, 2902.2150879, -2417.3447266, 3142.3837891, -5376.7397461, 5319.5595703
4: -2141.9882812, 3125.6911621, -2316.7839355, 3386.2197266, -5528.2080078, 5442.4750977

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5782167, upper bound: 4974.5839341
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5782154, upper bound: 4974.5812034
time: 1.58 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3140.6232910, 3731.9343262, -2594.8818359, 3088.5534668, -6229.1767578, 6326.8154297
1: -2640.0566406, 2988.4724121, -2187.3693848, 2470.8066406, -5110.8627930, 5175.8413086
2: -2150.2568359, 2951.0505371, -1777.1563721, 2443.1789551, -4593.2431641, 4728.2070312
3: -2879.3957520, 3736.6809082, -2379.3312988, 3091.6569824, -5971.0522461, 6116.0117188
4: -2763.7485352, 4023.0593262, -2280.6594238, 3331.5805664, -6095.3291016, 6303.7187500

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5652650, upper bound: 4974.5591721
time: 1.49 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5801255, upper bound: 4974.5797807
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3133.0766602, 3723.7526855, -2614.0410156, 3114.5356445, -6247.6123047, 6337.7919922
1: -2633.9968262, 2981.9306641, -2204.3535156, 2491.1259766, -5125.1225586, 5186.2836914
2: -2145.2973633, 2944.6452637, -1790.9177246, 2463.4550781, -4608.5546875, 4735.5625000
3: -2872.9260254, 3728.5656738, -2398.1857910, 3117.3151855, -5990.2412109, 6126.7514648
4: -2757.4792480, 4014.3322754, -2298.6118164, 3358.9860840, -6116.4653320, 6312.9443359

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5780953, upper bound: 4974.5821537
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5780823, upper bound: 4974.5791336
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2461.8269043, 2926.8020020, -2963.7370605, 3493.6325684, -5955.4594727, 5890.5385742
1: -2073.5930176, 2341.1269531, -2486.5678711, 2796.9401855, -4870.5332031, 4827.6943359
2: -1685.2219238, 2313.9665527, -2023.2659912, 2759.2092285, -4444.4301758, 4336.7846680
3: -2255.2331543, 2929.0500488, -2699.3117676, 3499.6494141, -5754.8828125, 5628.3618164
4: -2162.1079102, 3154.4299316, -2593.7961426, 3760.4748535, -5922.5815430, 5748.2260742

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5561653, upper bound: 4974.5595324
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611631, upper bound: 4974.5696001
time: 1.30 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2461.8269043, 2926.8020020, -3588.3063965, 4236.6118164, -6698.4384766, 6510.7236328
1: -2073.5930176, 2341.1269531, -3007.4550781, 3394.5661621, -5468.1591797, 5348.5815430
2: -1685.2219238, 2313.9665527, -2451.4675293, 3347.7685547, -5032.9902344, 4762.0561523
3: -2255.2331543, 2929.0500488, -3275.0075684, 4243.7778320, -6497.4672852, 6204.0576172
4: -2162.1079102, 3154.4299316, -3148.2717285, 4563.0805664, -6725.1879883, 6299.5239258

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5561653, upper bound: 4974.5595324
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611631, upper bound: 4974.5696001
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3155.8132324, 3749.9401855, -2963.7370605, 3493.6325684, -6647.0053711, 6709.9287109
1: -2652.8469238, 3002.8391113, -2486.5678711, 2796.9401855, -5449.7866211, 5489.0214844
2: -2160.6340332, 2965.2431641, -2023.2659912, 2759.2092285, -4918.8969727, 4987.2485352
3: -2893.2514648, 3754.6979980, -2699.3117676, 3499.6494141, -6391.7934570, 6454.0097656
4: -2777.0266113, 4042.4172363, -2593.7961426, 3760.4748535, -6537.5014648, 6635.9506836

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
time: 1.29 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3155.8132324, 3749.9401855, -3588.3063965, 4236.6118164, -7386.1166992, 7329.7338867
1: -2652.8469238, 3002.8391113, -3007.4550781, 3394.5661621, -6045.2670898, 6008.2080078
2: -2160.6340332, 2965.2431641, -2451.4675293, 3347.7685547, -5506.6777344, 5412.5200195
3: -2893.2514648, 3754.6979980, -3275.0075684, 4243.7778320, -7133.5585938, 7027.8720703
4: -2777.0266113, 4042.4172363, -3148.2717285, 4563.0805664, -7339.7207031, 7186.3525391

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
time: 1.14 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4974.5611485, upper bound: 4974.5677533
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2448.2446289, 2908.0886230, -2607.4826660, 3108.3835449, -5556.6274414, 5515.5708008
1: -2062.9040527, 2327.0476074, -2199.6020508, 2486.7866211, -4549.6904297, 4526.6494141
2: -1675.8500977, 2299.6887207, -1786.9512939, 2459.6972656, -4135.5468750, 4086.6398926
3: -2242.1157227, 2911.2124023, -2393.8762207, 3111.3488770, -5353.4648438, 5305.0888672
4: -2149.4912109, 3135.0244141, -2293.7639160, 3354.7072754, -5504.1982422, 5428.7880859

Time for backsubstitution: 1.49 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.64 + 416.32 = 420.96 seconds
