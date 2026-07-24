## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 4810.657341545514


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062)
1: (-294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011)
2: (-466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004)
3: (-542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918)
4: (-407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.79 + 1.91 = 4.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -4810.7054486, upper bound: 4810.7054486

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.7014154, upper bound: 4810.7009379
time: 0.73 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6997118, upper bound: 4810.6997118
time: 0.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.57 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -4810.7014154, upper bound: 4810.7009379
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -4810.6997118, upper bound: 4810.6997118

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2436.2365723, 2667.9499512, -2668.5781250, 2909.0441895, -5345.2797852, 5336.5283203
1: -268.5873413, 185.8037109, -293.0661926, 203.8444366, -472.4317627, 478.8699036
2: -424.1188354, 500.7198486, -464.0358582, 546.4890747, -970.6079102, 964.7556763
3: -494.4094238, 313.4508972, -540.0715942, 342.5705566, -836.9799805, 853.5224609
4: -370.5297546, 404.5379639, -405.1204834, 441.4743042, -812.0040283, 809.6583862

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6938567, upper bound: 4810.6952182
time: 0.63 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.7007233, upper bound: 4810.7007283
time: 0.62 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2603.2861328, 2847.8293457, -2682.2661133, 2923.5505371, -5526.8364258, 5530.0957031
1: -286.7768250, 198.7184753, -294.5390930, 204.9020233, -491.6788330, 493.2575684
2: -453.2702637, 534.8124390, -466.4013367, 549.2590942, -1002.5293579, 1001.2136230
3: -527.9588623, 335.0007019, -542.8253784, 344.3062134, -872.2650757, 877.8260498
4: -395.7932434, 432.1193542, -407.1929321, 443.7026672, -839.4959106, 839.3122559

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4809.9523190, upper bound: 4810.3483493
time: 0.60 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6996785, upper bound: 4810.6996785
time: 0.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.03 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 0, lower bound: -4810.6938567, upper bound: 4810.6952182
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 0, lower bound: -4810.7007233, upper bound: 4810.7007283
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 4.03
Output dim: 0, lower bound: -4809.9523190, upper bound: 4810.3483493
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 0, lower bound: -4810.6996785, upper bound: 4810.6996785

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -2364.0988770, 2594.1391602, -2662.4514160, 2902.6157227, -5266.7148438, 5256.5903320
1: -261.0791016, 180.1004486, -292.4105530, 203.3617859, -464.4408875, 472.5109863
2: -411.8400879, 486.2289734, -462.9835815, 545.2391968, -957.0792847, 949.2125244
3: -480.1896667, 304.5002136, -538.8368530, 341.7958069, -821.9854736, 843.3369141
4: -359.8471680, 392.9655151, -404.1928406, 440.4803467, -800.3274536, 797.1582642

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6720648, upper bound: 4810.6640650
time: 0.66 seconds

## Relational analysis of NS_A1_A1_A2

### Relational analysis result of NS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6883731, upper bound: 4810.6857974
time: 0.61 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -2411.5678711, 2644.0168457, -2668.5781250, 2909.0441895, -5320.6108398, 5312.5947266
1: -266.1227417, 183.8573456, -293.0661926, 203.8444366, -469.9671631, 476.9235229
2: -419.9867859, 496.0987854, -464.0358582, 546.4890747, -966.4758301, 960.1346436
3: -489.6696472, 310.4970093, -540.0715942, 342.5705566, -832.2401733, 850.5686035
4: -366.9079590, 400.8459778, -405.1204834, 441.4743042, -808.3822632, 805.9664307

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_A1

### Relational analysis result of NS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.7005065, upper bound: 4810.6991765
time: 0.65 seconds

## Relational analysis of NS_A1_A2_A2

### Relational analysis result of NS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.7007233, upper bound: 4810.7007018
time: 0.69 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -2603.2861328, 2847.8293457, -2675.1577148, 2916.2265625, -5519.5122070, 5522.9873047
1: -286.7768250, 198.7184753, -293.7941284, 204.3429260, -491.1197510, 492.5126038
2: -453.2702637, 534.8124390, -465.1870728, 547.8630371, -1001.1332397, 999.9992065
3: -527.9588623, 335.0007019, -541.4212646, 343.4216309, -871.3804321, 876.4219971
4: -395.7932434, 432.1193542, -406.1298218, 442.5853577, -838.3786011, 838.2491455

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6691269, upper bound: 4810.6660161
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6992864, upper bound: 4810.6992864
time: 1.40 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.93 seconds
NS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.93
Output dim: 0, lower bound: -4810.6720648, upper bound: 4810.6640650
NS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.93
Output dim: 0, lower bound: -4810.6883731, upper bound: 4810.6857974
NS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.93
Output dim: 0, lower bound: -4810.7005065, upper bound: 4810.6991765
NS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.93
Output dim: 0, lower bound: -4810.7007233, upper bound: 4810.7007018
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.93
Output dim: 0, lower bound: -4810.6691269, upper bound: 4810.6660161
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.93
Output dim: 0, lower bound: -4810.6992864, upper bound: 4810.6992864

## BFS NS instance: NS_A1_A1_A1

### Backsubstitution after applying NS history:
0: -1777.4074707, 2029.1104736, -2352.4692383, 2605.9101562, -4383.3173828, 4381.5795898
1: -203.2941895, 134.3316345, -262.1099548, 179.1908569, -382.4849854, 396.4415894
2: -311.8141174, 380.3331909, -411.1121216, 489.7107849, -801.5249023, 791.4453125
3: -368.4182434, 236.4111633, -481.2287903, 305.6204224, -674.0385132, 717.6398926
4: -273.0012207, 307.5889587, -359.4563599, 395.6164551, -668.6174927, 667.0452271

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_A1_B1

### Relational analysis result of NS_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6485296, upper bound: 4810.6482933
time: 0.67 seconds

## Relational analysis of NS_A1_A1_A1_B2

### Relational analysis result of NS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6485296, upper bound: 4810.6640650
time: 0.65 seconds

## BFS NS instance: NS_A1_A1_A2

### Backsubstitution after applying NS history:
0: -3113.6955566, 3411.8327637, -2586.2031250, 2823.3271484, -5937.0224609, 5998.0361328
1: -343.7941895, 237.0987854, -284.2121582, 197.5654907, -541.3596802, 521.3109131
2: -542.7215576, 639.8481445, -449.1849365, 530.5197144, -1073.2412109, 1089.0330811
3: -634.9635010, 401.3323669, -522.8811035, 332.0360718, -966.9995728, 924.2135010
4: -476.3038330, 517.1386108, -392.1173706, 428.4439087, -904.7477417, 909.2559814

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_A2_B1

### Relational analysis result of NS_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6619920, upper bound: 4810.6673415
time: 0.65 seconds

## Relational analysis of NS_A1_A1_A2_B2

### Relational analysis result of NS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6619920, upper bound: 4810.6857974
time: 0.67 seconds

## BFS NS instance: NS_A1_A2_A1

### Backsubstitution after applying NS history:
0: -2346.9379883, 2567.9526367, -2663.8017578, 2904.3366699, -5251.2744141, 5231.7543945
1: -258.4011841, 178.9010162, -292.5847473, 203.4708252, -461.8720093, 471.4857483
2: -408.6872559, 481.7813416, -463.2310791, 545.5988770, -954.2861328, 945.0123901
3: -475.7019653, 301.7107849, -539.1566772, 341.9964294, -817.6983643, 840.8674316
4: -356.6854248, 389.5185242, -404.4181519, 440.7588806, -797.4442749, 793.9366455

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A1_A1

### Relational analysis result of NS_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6622560, upper bound: 4810.6571886
time: 0.63 seconds

## Relational analysis of NS_A1_A2_A1_A2

### Relational analysis result of NS_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6960852, upper bound: 4810.6914751
time: 0.60 seconds

## BFS NS instance: NS_A1_A2_A2

### Backsubstitution after applying NS history:
0: -2379.6652832, 2611.3449707, -2668.5781250, 2909.0441895, -5288.7089844, 5279.9228516
1: -262.7987976, 181.3813629, -293.0661926, 203.8444366, -466.6432190, 474.4475403
2: -414.5733032, 489.8923035, -464.0358582, 546.4890747, -961.0623779, 953.9279785
3: -483.4346008, 306.5720520, -540.0715942, 342.5705566, -826.0051270, 846.6436768
4: -362.1672668, 395.8703003, -405.1204834, 441.4743042, -803.6415405, 800.9907227

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A2_B1

### Relational analysis result of NS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6709238, upper bound: 4810.6703278
time: 0.69 seconds

## Relational analysis of NS_A1_A2_A2_B2

### Relational analysis result of NS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6965809, upper bound: 4810.6933438
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2497.5078125, 2736.3771973, -2668.9860840, 2909.7800293, -5407.2880859, 5405.3623047
1: -275.4814148, 190.3940125, -293.1359558, 203.8590393, -479.3404541, 483.5299377
2: -435.1302185, 513.2310181, -464.1275635, 546.6090698, -981.7392578, 977.3585815
3: -506.9050293, 321.6378479, -540.1751099, 342.6445618, -849.5495605, 861.8129883
4: -379.9759521, 414.8594055, -405.1937256, 441.5881958, -821.5641479, 820.0531006

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6691269, upper bound: 4810.6660161
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6691269, upper bound: 4810.6660161
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2586.9873047, 2831.5256348, -2675.1577148, 2916.2265625, -5503.2138672, 5506.6835938
1: -285.1144409, 197.4244385, -293.7941284, 204.3429260, -489.4573669, 491.2185669
2: -450.5158081, 531.6378784, -465.1870728, 547.8630371, -998.3788452, 996.8248901
3: -524.7919922, 333.0097961, -541.4212646, 343.4216309, -868.2136230, 874.4310303
4: -393.3863831, 429.5817566, -406.1298218, 442.5853577, -835.9717407, 835.7115479

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6984012, upper bound: 4810.6982236
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6992864, upper bound: 4810.6992864
time: 0.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.13 seconds
NS_A1_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.13
Output dim: 0, lower bound: -4810.6485296, upper bound: 4810.6482933
NS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 0, lower bound: -4810.6485296, upper bound: 4810.6640650
NS_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 0, lower bound: -4810.6619920, upper bound: 4810.6673415
NS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 0, lower bound: -4810.6619920, upper bound: 4810.6857974
NS_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 0, lower bound: -4810.6622560, upper bound: 4810.6571886
NS_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 0, lower bound: -4810.6960852, upper bound: 4810.6914751
NS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 0, lower bound: -4810.6709238, upper bound: 4810.6703278
NS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 0, lower bound: -4810.6965809, upper bound: 4810.6933438
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 0, lower bound: -4810.6691269, upper bound: 4810.6660161
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 0, lower bound: -4810.6691269, upper bound: 4810.6660161
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 0, lower bound: -4810.6984012, upper bound: 4810.6982236
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 0, lower bound: -4810.6992864, upper bound: 4810.6992864

## BFS NS instance: NS_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -1777.4074707, 2029.1104736, -3402.4357910, 3711.8159180, -5489.2236328, 5431.5463867
1: -203.2941895, 134.3316345, -374.0161133, 259.7945251, -463.0887146, 508.3477478
2: -311.8141174, 380.3331909, -592.4179688, 697.2626953, -1009.0767822, 972.7511597
3: -368.4182434, 236.4111633, -691.4116821, 437.4001465, -805.8183594, 927.8227539
4: -273.0012207, 307.5889587, -519.0817871, 563.4088135, -836.4098511, 826.6707764

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A1_B2_A1

### Relational analysis result of NS_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6478743, upper bound: 4810.6640649
time: 0.68 seconds

## Relational analysis of NS_A1_A1_A1_B2_A2

### Relational analysis result of NS_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6426198, upper bound: 4810.6575588
time: 0.66 seconds

## BFS NS instance: NS_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -3111.5339355, 3409.7875977, -2042.0007324, 2300.4653320, -5411.9985352, 5451.7880859
1: -343.5803833, 236.9325256, -230.9221497, 154.8239594, -498.4042664, 467.8546753
2: -542.3686523, 639.4556885, -357.5017700, 432.6388855, -975.0074463, 996.9574585
3: -634.5559082, 401.0732422, -420.8696899, 269.2009888, -903.7568970, 821.9428711
4: -475.9874878, 516.8262329, -312.5304871, 349.6065674, -825.5940552, 829.3566895

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A2_B1_A1

### Relational analysis result of NS_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6614472, upper bound: 4810.6627782
time: 0.61 seconds

## Relational analysis of NS_A1_A1_A2_B1_A2

### Relational analysis result of NS_A1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6502600, upper bound: 4810.6627782
time: 0.63 seconds

## BFS NS instance: NS_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3113.7250977, 3411.8608398, -3414.6606445, 3723.2343750, -6836.9594727, 6826.5214844
1: -343.7971802, 237.1010590, -375.1936340, 260.7222595, -604.5194092, 612.2946777
2: -542.7265015, 639.8535156, -594.4878540, 699.4266357, -1242.1530762, 1234.3413086
3: -634.9691162, 401.3358765, -693.6494141, 438.8188782, -1073.7879639, 1094.9853516
4: -476.3081360, 517.1430054, -520.8931885, 565.1420288, -1041.4500732, 1038.0360107

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A2_B2_B1

### Relational analysis result of NS_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6619920, upper bound: 4810.6857974
time: 0.65 seconds

## Relational analysis of NS_A1_A1_A2_B2_B2

### Relational analysis result of NS_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6619920, upper bound: 4810.6857974
time: 0.59 seconds

## BFS NS instance: NS_A1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -1764.0119629, 2005.6333008, -2354.1616211, 2607.8339844, -4371.8452148, 4359.7949219
1: -200.9374390, 133.4766388, -262.3071899, 179.3291321, -380.2665405, 395.7838135
2: -309.2850647, 376.5920410, -411.4072571, 490.1191711, -799.4041748, 787.9992676
3: -364.7510681, 233.9407959, -481.6125488, 305.8525696, -670.6035767, 715.5533447
4: -270.4344177, 304.6105957, -359.7238159, 395.9314270, -666.3657227, 664.3344116

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A1_A1_B1

### Relational analysis result of NS_A1_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6383583, upper bound: 4810.6370394
time: 0.61 seconds

## Relational analysis of NS_A1_A2_A1_A1_B2

### Relational analysis result of NS_A1_A2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6383583, upper bound: 4810.6571886
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -3090.8012695, 3379.2165527, -2587.9082031, 2825.4682617, -5916.2695312, 5967.1240234
1: -340.4849854, 235.5458527, -284.4309692, 197.7069550, -538.1918335, 519.9767456
2: -538.4212036, 634.2402344, -449.4780884, 530.9713135, -1069.3925781, 1083.7181396
3: -629.1409302, 397.7811890, -523.2432251, 332.2901611, -961.4310913, 921.0244141
4: -472.1989746, 512.7055664, -392.3713379, 428.7931824, -900.9920044, 905.0767822

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A1_A2_B1

### Relational analysis result of NS_A1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6672813, upper bound: 4810.6655902
time: 0.69 seconds

## Relational analysis of NS_A1_A2_A1_A2_B2

### Relational analysis result of NS_A1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6672813, upper bound: 4810.6914751
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -2076.8251953, 2323.4047852, -2047.8309326, 2306.4934082, -4383.3183594, 4371.2358398
1: -233.3775940, 157.7392578, -231.5381622, 155.2848358, -388.6623840, 389.2774048
2: -363.8680725, 435.8665771, -358.5087280, 433.8191833, -797.6872559, 794.3751831
3: -427.0780640, 271.5670166, -422.0451660, 269.9289551, -697.0070190, 693.6121216
4: -318.3768921, 352.2623291, -313.4083252, 350.5456848, -668.9224854, 665.6706543

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A2_B1_A1

### Relational analysis result of NS_A1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6539321, upper bound: 4810.6495673
time: 0.66 seconds

## Relational analysis of NS_A1_A2_A2_B1_A2

### Relational analysis result of NS_A1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6539321, upper bound: 4810.6703278
time: 0.62 seconds

## BFS NS instance: NS_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2302.4077148, 2529.9978027, -3421.0183105, 3729.8425293, -6032.2495117, 5951.0136719
1: -254.3620300, 175.4580383, -375.8654785, 261.2221069, -515.5841064, 551.3234863
2: -400.6888123, 474.7318420, -595.5864868, 700.7016602, -1101.3905029, 1070.3182373
3: -467.4345398, 296.5421143, -694.9254761, 439.6126404, -907.0471191, 991.4675903
4: -350.1211243, 383.4837646, -521.8579712, 566.1584473, -916.2795410, 905.3417358

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A2_B2_A1

### Relational analysis result of NS_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6771562, upper bound: 4810.6683489
time: 0.65 seconds

## Relational analysis of NS_A1_A2_A2_B2_A2

### Relational analysis result of NS_A1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6771562, upper bound: 4810.6933438
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2497.5078125, 2736.3771973, -2425.0693359, 2656.5424805, -5154.0502930, 5161.4458008
1: -275.4814148, 190.3940125, -267.4217224, 184.9262695, -460.4076843, 457.8157349
2: -435.1302185, 513.2310181, -422.2095032, 498.5169678, -933.6472168, 935.4405518
3: -506.9050293, 321.6378479, -492.1878052, 312.0658264, -818.9708252, 813.8255615
4: -379.9759521, 414.8594055, -368.8513184, 402.7807007, -782.7566528, 783.7106934

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6621943, upper bound: 4810.6586191
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586677
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2497.5078125, 2736.3771973, -2589.8608398, 2834.0380859, -5331.5458984, 5326.2377930
1: -275.4814148, 190.3940125, -285.3700867, 197.6663055, -473.1477051, 475.7640991
2: -435.1302185, 513.2310181, -450.9697876, 532.1570435, -967.2872314, 964.2007446
3: -506.9050293, 321.6378479, -525.2789917, 333.3348083, -840.2398682, 846.9168091
4: -379.9759521, 414.8594055, -393.7722168, 429.9987793, -809.9747314, 808.6315918

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6621943, upper bound: 4810.6586191
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586677
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -2536.7250977, 2769.7985840, -2670.3293457, 2911.5014648, -5448.2255859, 5440.1279297
1: -278.8332520, 193.5720825, -293.3101807, 203.9672089, -482.8004456, 486.8822632
2: -441.5703430, 520.0559692, -464.3755188, 546.9686890, -988.5390625, 984.4315186
3: -513.4219971, 325.9767761, -540.4964600, 342.8453064, -856.2672119, 866.4732666
4: -385.1441040, 420.4842834, -405.4202881, 441.8668518, -827.0109253, 825.9045410

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_A1_A1

### Relational analysis result of NS_A2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6554438, upper bound: 4810.6544487
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_A1_A2

### Relational analysis result of NS_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6903025, upper bound: 4810.6889653
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -2552.1171875, 2795.9394531, -2675.1577148, 2916.2265625, -5468.3437500, 5471.0957031
1: -281.4902649, 194.7333679, -293.7941284, 204.3429260, -485.8331909, 488.5274963
2: -444.6107788, 524.8920898, -465.1870728, 547.8630371, -992.4737549, 990.0791016
3: -517.9617310, 328.7424927, -541.4212646, 343.4216309, -861.3833618, 870.1637573
4: -388.2098999, 424.1652222, -406.1298218, 442.5853577, -830.7952881, 830.2950439

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6992864, upper bound: 4810.6992864
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6992864, upper bound: 4810.6992864
time: 0.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.29 seconds
NS_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6478743, upper bound: 4810.6640649
NS_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6426198, upper bound: 4810.6575588
NS_A1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6614472, upper bound: 4810.6627782
NS_A1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6502600, upper bound: 4810.6627782
NS_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6619920, upper bound: 4810.6857974
NS_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6619920, upper bound: 4810.6857974
NS_A1_A2_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6383583, upper bound: 4810.6370394
NS_A1_A2_A1_A1_B2, status: Status.VERIFIED, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6383583, upper bound: 4810.6571886
NS_A1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6672813, upper bound: 4810.6655902
NS_A1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6672813, upper bound: 4810.6914751
NS_A1_A2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6539321, upper bound: 4810.6495673
NS_A1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6539321, upper bound: 4810.6703278
NS_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6771562, upper bound: 4810.6683489
NS_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6771562, upper bound: 4810.6933438
NS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6621943, upper bound: 4810.6586191
NS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586677
NS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6621943, upper bound: 4810.6586191
NS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586677
NS_A2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6554438, upper bound: 4810.6544487
NS_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6903025, upper bound: 4810.6889653
NS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6992864, upper bound: 4810.6992864
NS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 0, lower bound: -4810.6992864, upper bound: 4810.6992864

## BFS NS instance: NS_A1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1513.3447266, 1739.6315918, -3304.6359863, 3602.7548828, -5116.0986328, 5044.2675781
1: -173.8506622, 113.8424225, -362.9093018, 252.1116180, -425.9622498, 476.7517090
2: -266.0104980, 324.3844299, -575.4055176, 676.0224609, -942.0328979, 899.7898560
3: -314.6069031, 201.9948425, -670.9059448, 424.5046997, -739.1113892, 872.9006348
4: -233.1253204, 263.1195679, -504.0620728, 546.6350708, -779.7603760, 767.1816406

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A1_B2_A1_B1

### Relational analysis result of NS_A1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6632793, upper bound: 4810.6597850
time: 1.05 seconds

## Relational analysis of NS_A1_A1_A1_B2_A1_B2

### Relational analysis result of NS_A1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6605612, upper bound: 4810.6510653
time: 0.66 seconds

## BFS NS instance: NS_A1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1740.0871582, 1988.4365234, -3401.2033691, 3710.5541992, -5450.6401367, 5389.6396484
1: -199.1485901, 131.4482574, -373.8872986, 259.6984253, -458.8470154, 505.3355713
2: -305.3110352, 372.4704285, -592.2087402, 697.0205078, -1002.3315430, 964.6790161
3: -360.7361450, 231.5633087, -691.1721802, 437.2470093, -797.9830933, 922.7354126
4: -267.3030701, 301.3238831, -518.8986206, 563.2153320, -830.5184326, 820.2225342

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_A1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6401116, upper bound: 4810.6408175
time: 0.57 seconds

## Relational analysis of NS_A1_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6619450, upper bound: 4810.6553465
time: 0.68 seconds

## BFS NS instance: NS_A1_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3054.7121582, 3350.7397461, -2042.0007324, 2300.4653320, -5355.1777344, 5392.7402344
1: -337.5106812, 232.5603180, -230.9221497, 154.8239594, -492.3345337, 463.4824829
2: -532.4680176, 628.2628174, -357.5017700, 432.6388855, -965.1068726, 985.7645874
3: -623.0191650, 393.9740601, -420.8696899, 269.2009888, -892.2201538, 814.8436890
4: -467.2582092, 507.8631592, -312.5304871, 349.6065674, -816.8647461, 820.3936768

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A2_B1_A1_B1

### Relational analysis result of NS_A1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
time: 0.60 seconds

## Relational analysis of NS_A1_A1_A2_B1_A1_B2

### Relational analysis result of NS_A1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
time: 0.66 seconds

## BFS NS instance: NS_A1_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3594.1689453, 3908.4052734, -2021.7886963, 2278.4084473, -5872.5771484, 5930.1938477
1: -396.2433777, 273.2323914, -228.6797180, 153.2774200, -549.5208130, 501.9121094
2: -626.8465576, 736.8020020, -353.9935913, 428.4308167, -1055.2773438, 1090.7956543
3: -738.7770386, 462.4765320, -416.7322693, 266.5897522, -1005.3666382, 879.2087402
4: -553.5794678, 593.2048340, -309.4563293, 346.2433777, -899.8228149, 902.6611328

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A2_B1_A2_B1

### Relational analysis result of NS_A1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
time: 0.66 seconds

## Relational analysis of NS_A1_A1_A2_B1_A2_B2

### Relational analysis result of NS_A1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
time: 0.62 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -3113.7250977, 3411.8608398, -3182.7202148, 3484.0463867, -6597.7714844, 6594.5810547
1: -343.7971802, 237.1010590, -351.0711670, 242.6019897, -586.3991699, 588.1722412
2: -542.7265015, 639.8535156, -554.6397705, 653.9091797, -1196.6356201, 1194.4932861
3: -634.9691162, 401.3358765, -648.5881958, 410.0046387, -1044.9736328, 1049.9240723
4: -476.3081360, 517.1430054, -486.5436096, 528.3967285, -1004.7048340, 1003.6866455

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6812361, upper bound: 4810.6732626
time: 0.66 seconds

## Relational analysis of NS_A1_A1_A2_B2_B1_B2

### Relational analysis result of NS_A1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6837092, upper bound: 4810.6764889
time: 0.64 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3113.7250977, 3411.8608398, -3354.6970215, 3665.6115723, -6779.3369141, 6766.5576172
1: -343.7971802, 237.1010590, -369.3366089, 256.0509949, -599.8481445, 606.4376831
2: -542.7265015, 639.8535156, -584.2253418, 688.5413818, -1231.2677002, 1224.0788574
3: -634.9691162, 401.3358765, -682.4005127, 431.7973938, -1066.7664795, 1083.7363281
4: -476.3081360, 517.1430054, -511.9971924, 556.3627930, -1032.6706543, 1029.1400146

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6812361, upper bound: 4810.6732626
time: 0.66 seconds

## Relational analysis of NS_A1_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6837092, upper bound: 4810.6764889
time: 0.68 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -3088.5427246, 3377.1169434, -2043.8939209, 2302.4982910, -5391.0410156, 5421.0107422
1: -340.2636719, 235.3730011, -231.1320343, 154.9742584, -495.2379150, 466.5050354
2: -538.0466919, 633.8355103, -357.8244324, 433.0692139, -971.1159058, 991.6598511
3: -628.7091675, 397.5143738, -421.2686462, 269.4492188, -898.1583862, 818.7829590
4: -471.8632202, 512.3830566, -312.8159790, 349.9426270, -821.8057251, 825.1988525

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_A1_A2_B1_A1

### Relational analysis result of NS_A1_A2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6523277, upper bound: 4810.6363666
time: 0.69 seconds

## Relational analysis of NS_A1_A2_A1_A2_B1_A2

### Relational analysis result of NS_A1_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6664393, upper bound: 4810.6589634
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3090.8012695, 3379.2165527, -3415.8754883, 3724.7150879, -6815.5161133, 6795.0913086
1: -340.4849854, 235.5458527, -375.3447266, 260.8246765, -601.3095703, 610.8905640
2: -538.4212036, 634.2402344, -594.7048340, 699.7423706, -1238.1635742, 1228.9450684
3: -629.1409302, 397.7811890, -693.9365234, 438.9920654, -1068.1330566, 1091.7177734
4: -472.1989746, 512.7055664, -521.0873413, 565.3855591, -1037.5844727, 1033.7928467

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6672813, upper bound: 4810.6914751
time: 0.66 seconds

## Relational analysis of NS_A1_A2_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6672813, upper bound: 4810.6914751
time: 0.62 seconds

## BFS NS instance: NS_A1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3128.8046875, 3430.0703125, -2047.8309326, 2306.4934082, -5435.2978516, 5477.9013672
1: -345.5538330, 238.4068298, -231.5381622, 155.2848358, -500.8386841, 469.9449463
2: -545.3799438, 643.6171265, -358.5087280, 433.8191833, -979.1990356, 1002.1258545
3: -638.0902100, 403.4536743, -422.0451660, 269.9289551, -908.0191650, 825.4988403
4: -478.5637512, 520.1351318, -313.4083252, 350.5456848, -829.1092529, 833.5434570

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_A2_B1_A2_A1

### Relational analysis result of NS_A1_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6533802, upper bound: 4810.6651342
time: 0.56 seconds

## Relational analysis of NS_A1_A2_A2_B1_A2_A2

### Relational analysis result of NS_A1_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6425393, upper bound: 4810.6645738
time: 0.66 seconds

## BFS NS instance: NS_A1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1798.0582275, 2051.3286133, -3418.6442871, 3727.6340332, -5525.6904297, 5469.9716797
1: -205.5346680, 136.0364532, -375.6323242, 261.0393982, -466.5740662, 511.6687317
2: -315.3123474, 385.0217896, -595.1964722, 700.2734985, -1015.5857544, 980.2182617
3: -372.5011292, 239.1112976, -694.4710083, 439.3311462, -811.8322754, 933.5822754
4: -275.9544983, 311.2905884, -521.5119019, 565.8191528, -841.7736816, 832.8023071

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_A2_B2_A1_A1

### Relational analysis result of NS_A1_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6534987, upper bound: 4810.6683400
time: 0.68 seconds

## Relational analysis of NS_A1_A2_A2_B2_A1_A2

### Relational analysis result of NS_A1_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6531685, upper bound: 4810.6670858
time: 0.62 seconds

## BFS NS instance: NS_A1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3133.2282715, 3434.1193848, -3421.0183105, 3729.8425293, -6863.0683594, 6855.1357422
1: -345.9767761, 238.7386017, -375.8654785, 261.2221069, -607.1988525, 614.6040649
2: -546.1246338, 644.4014893, -595.5864868, 700.7016602, -1246.8262939, 1239.9877930
3: -638.9020996, 403.9585571, -694.9254761, 439.6126404, -1078.5146484, 1098.8839111
4: -479.1801758, 520.7603149, -521.8579712, 566.1584473, -1045.3386230, 1042.6182861

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_A2_B2_A2_B1

### Relational analysis result of NS_A1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6539321, upper bound: 4810.6933438
time: 0.62 seconds

## Relational analysis of NS_A1_A2_A2_B2_A2_B2

### Relational analysis result of NS_A1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6539321, upper bound: 4810.6933438
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -2408.5551758, 2636.0456543, -2063.5051270, 2266.3430176, -4674.8984375, 4699.5502930
1: -265.2778625, 183.4703979, -227.6954803, 156.8362885, -422.1141357, 411.1658325
2: -419.5198975, 493.8172607, -359.4161072, 423.7663269, -843.2861938, 853.2333374
3: -488.1751709, 309.7917175, -418.4036560, 265.5219421, -753.6971436, 728.1953125
4: -366.2025757, 399.4705505, -313.7945251, 343.1575317, -709.3600464, 713.2650757

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586191
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586191
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -2496.3740234, 2735.1936035, -2379.4299316, 2609.7792969, -5106.1533203, 5114.6235352
1: -275.3605957, 190.3049011, -262.6293945, 181.3446045, -456.7051392, 452.9342651
2: -434.9380798, 513.0031128, -414.5296021, 489.5097961, -924.4478760, 927.5327148
3: -506.6809998, 321.4952087, -483.2355042, 306.4008789, -813.0817871, 804.7307129
4: -379.8080750, 414.6780090, -362.1143799, 395.6056824, -775.4137573, 776.7923584

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586677
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586677
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -2408.5551758, 2636.0456543, -2198.3356934, 2411.4101562, -4819.9653320, 4834.3808594
1: -265.2778625, 183.4703979, -242.3534088, 167.2452545, -432.5231323, 425.8237000
2: -419.5198975, 493.8172607, -382.9600220, 451.3033142, -870.8231812, 876.7772827
3: -488.1751709, 309.7917175, -445.4536438, 282.9427185, -771.1177979, 755.2453003
4: -366.2025757, 399.4705505, -334.1362915, 365.4711914, -731.6737671, 733.6068115

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.2917383, upper bound: 4810.1246228
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6573729, upper bound: 4810.6555752
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -2496.3740234, 2735.1936035, -2552.9064941, 2795.0869141, -5291.4609375, 5288.1000977
1: -275.3605957, 190.3049011, -281.4103699, 194.7635498, -470.1241150, 471.7152405
2: -434.9380798, 513.0031128, -444.7479248, 524.6442871, -959.5823975, 957.7509766
3: -506.6809998, 321.4952087, -518.0458984, 328.6477051, -835.3286133, 839.5411377
4: -379.8080750, 414.6780090, -388.3470154, 424.0177917, -803.8258667, 803.0250244

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586677
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586677
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -3278.8442383, 3575.5529785, -2594.5461426, 2832.8117676, -6111.6562500, 6170.0991211
1: -360.2859497, 250.2220306, -285.1742249, 198.2246857, -558.5104980, 535.3961792
2: -570.7663574, 671.6951294, -450.5863953, 532.3973999, -1103.1636963, 1122.2808838
3: -665.8966064, 421.4064636, -524.5136108, 333.1674194, -999.0640259, 945.9200439
4: -499.9883728, 542.9427490, -393.3103333, 429.9407349, -929.9290771, 936.2530518

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6611105, upper bound: 4810.6601476
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6611105, upper bound: 4810.6889653
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -2552.1171875, 2795.9394531, -2430.3398438, 2661.9804688, -5214.0971680, 5226.2783203
1: -281.4902649, 194.7333679, -267.9778137, 185.3419800, -466.8322449, 462.7111816
2: -444.6107788, 524.8920898, -423.1104126, 499.5780640, -944.1888428, 948.0025024
3: -517.9617310, 328.7424927, -493.2385864, 312.7245483, -830.6862793, 821.9810791
4: -388.2098999, 424.1652222, -369.6414795, 403.6257019, -791.8355713, 793.8067017

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6688223, upper bound: 4810.6633988
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6905492, upper bound: 4810.6905492
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2552.1171875, 2795.9394531, -2596.0332031, 2840.4089355, -5392.5263672, 5391.9711914
1: -281.4902649, 194.7333679, -286.0213013, 198.1491394, -479.6393738, 480.7546692
2: -444.6107788, 524.8920898, -452.0323486, 533.3973999, -978.0080566, 976.9244385
3: -517.9617310, 328.7424927, -526.5275269, 334.1038818, -852.0656128, 855.2699585
4: -388.2098999, 424.1652222, -394.7090149, 430.9857483, -819.1956787, 818.8742676

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6688223, upper bound: 4810.6633988
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6905492, upper bound: 4810.6905492
time: 0.68 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.28 seconds
NS_A1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6632793, upper bound: 4810.6597850
NS_A1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6605612, upper bound: 4810.6510653
NS_A1_A1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6401116, upper bound: 4810.6408175
NS_A1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6619450, upper bound: 4810.6553465
NS_A1_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
NS_A1_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
NS_A1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
NS_A1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
NS_A1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6812361, upper bound: 4810.6732626
NS_A1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6837092, upper bound: 4810.6764889
NS_A1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6812361, upper bound: 4810.6732626
NS_A1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6837092, upper bound: 4810.6764889
NS_A1_A2_A1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6523277, upper bound: 4810.6363666
NS_A1_A2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6664393, upper bound: 4810.6589634
NS_A1_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6672813, upper bound: 4810.6914751
NS_A1_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6672813, upper bound: 4810.6914751
NS_A1_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6533802, upper bound: 4810.6651342
NS_A1_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6425393, upper bound: 4810.6645738
NS_A1_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6534987, upper bound: 4810.6683400
NS_A1_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6531685, upper bound: 4810.6670858
NS_A1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6539321, upper bound: 4810.6933438
NS_A1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6539321, upper bound: 4810.6933438
NS_A2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586191
NS_A2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586191
NS_A2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586677
NS_A2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586677
NS_A2_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.2917383, upper bound: 4810.1246228
NS_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6573729, upper bound: 4810.6555752
NS_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586677
NS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6617909, upper bound: 4810.6586677
NS_A2_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6611105, upper bound: 4810.6601476
NS_A2_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6611105, upper bound: 4810.6889653
NS_A2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6688223, upper bound: 4810.6633988
NS_A2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6905492, upper bound: 4810.6905492
NS_A2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6688223, upper bound: 4810.6633988
NS_A2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 0, lower bound: -4810.6905492, upper bound: 4810.6905492

## BFS NS instance: NS_A1_A1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1513.3447266, 1739.6315918, -3254.6169434, 3551.0695801, -5064.4140625, 4994.2470703
1: -173.8506622, 113.8424225, -357.6165771, 248.2691803, -422.1198425, 471.4589539
2: -266.0104980, 324.3844299, -566.7473145, 666.2456055, -932.2561035, 891.1315918
3: -314.6069031, 201.9948425, -660.8684082, 418.2927856, -732.8994141, 862.8632812
4: -233.1253204, 263.1195679, -496.4519653, 538.7908325, -771.9161377, 759.5715332

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_A1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6605612, upper bound: 4810.6510551
time: 0.67 seconds

## Relational analysis of NS_A1_A1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_A1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6605612, upper bound: 4810.6510653
time: 0.65 seconds

## BFS NS instance: NS_A1_A1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1499.3769531, 1724.0477295, -3638.3217773, 3941.7680664, -5441.1450195, 5362.3696289
1: -172.2761078, 112.7759018, -399.4946594, 276.9820251, -449.2581177, 512.0086670
2: -263.5783691, 321.4132080, -634.2475586, 743.2933350, -1006.8717041, 955.6607666
3: -311.7233276, 200.1635284, -745.5368652, 466.8203735, -778.5435181, 945.7003784
4: -230.9902496, 260.7449341, -559.0698242, 598.7907104, -829.7807617, 819.8145752

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_A1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6605612, upper bound: 4810.6510551
time: 0.62 seconds

## Relational analysis of NS_A1_A1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_A1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6605612, upper bound: 4810.6510653
time: 0.68 seconds

## BFS NS instance: NS_A1_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1724.0186768, 1971.2467041, -3339.3425293, 3646.1235352, -5370.1420898, 5310.5888672
1: -197.4126282, 130.2012329, -367.2818604, 254.9283447, -452.3409119, 497.4830627
2: -302.5283508, 369.2284546, -581.4436035, 684.8979492, -987.4261475, 950.6719360
3: -357.5392151, 229.5215454, -678.6553955, 429.5114136, -787.0506592, 908.1769409
4: -264.8898315, 298.7161865, -509.3836670, 553.4874878, -818.3773193, 808.0997314

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6613534, upper bound: 4810.6515113
time: 0.62 seconds

## Relational analysis of NS_A1_A1_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_A1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6613534, upper bound: 4810.6515112
time: 0.67 seconds

## BFS NS instance: NS_A1_A1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3054.6645508, 3350.6945801, -2004.6320801, 2261.4929199, -5316.1572266, 5355.3261719
1: -337.5058899, 232.5565796, -226.9620361, 151.9596863, -489.4655762, 459.5186157
2: -532.4600220, 628.2540283, -350.9580688, 425.3329468, -957.7929688, 979.2120361
3: -623.0100708, 393.9682312, -413.3220825, 264.5703735, -887.5804443, 807.2902832
4: -467.2511902, 507.8561401, -306.8354492, 343.7341309, -810.9853516, 814.6915894

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A2_B1_A1_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6614472, upper bound: 4810.6627782
time: 0.77 seconds

## Relational analysis of NS_A1_A1_A2_B1_A1_B1_B2

### Relational analysis result of NS_A1_A1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6614472, upper bound: 4810.6627782
time: 0.67 seconds

## BFS NS instance: NS_A1_A1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3054.8159180, 3350.8371582, -2371.4282227, 2640.5561523, -5695.3706055, 5722.2651367
1: -337.5208435, 232.5682373, -266.9859314, 179.6307831, -517.1516113, 499.5541077
2: -532.4846802, 628.2815552, -415.4385071, 499.4791565, -1031.9638672, 1043.7200928
3: -623.0386353, 393.9864197, -493.4077148, 311.0498047, -934.0884399, 887.3940430
4: -467.2734070, 507.8781128, -366.0463867, 401.6316223, -868.9049683, 873.9244385

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A2_B1_A1_B2_B1

### Relational analysis result of NS_A1_A1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6614472, upper bound: 4810.6627782
time: 0.66 seconds

## Relational analysis of NS_A1_A1_A2_B1_A1_B2_B2

### Relational analysis result of NS_A1_A1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6614472, upper bound: 4810.6627782
time: 0.64 seconds

## BFS NS instance: NS_A1_A1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3594.1169434, 3908.3571777, -2004.6320801, 2261.4929199, -5855.6098633, 5912.9882812
1: -396.2382812, 273.2283936, -226.9620361, 151.9596863, -548.1979980, 500.1904297
2: -626.8380127, 736.7926025, -350.9580688, 425.3329468, -1052.1708984, 1087.7507324
3: -738.7671509, 462.4703674, -413.3220825, 264.5703735, -1003.3375244, 875.7924194
4: -553.5717773, 593.1973877, -306.8354492, 343.7341309, -897.3059082, 900.0327759

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
time: 0.62 seconds

## Relational analysis of NS_A1_A1_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_A1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
time: 0.68 seconds

## BFS NS instance: NS_A1_A1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3594.3269043, 3908.5534668, -2371.4282227, 2640.5561523, -6234.8818359, 6279.9814453
1: -396.2588501, 273.2444458, -266.9859314, 179.6307831, -575.7379150, 540.2302856
2: -626.8724976, 736.8302002, -415.4385071, 499.4791565, -1126.3516846, 1152.2686768
3: -738.8065186, 462.4953613, -493.4077148, 311.0498047, -1049.8563232, 955.9030151
4: -553.6025391, 593.2274780, -366.0463867, 401.6316223, -955.2341309, 959.2738647

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A2_B1_A2_B2_B1

### Relational analysis result of NS_A1_A1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
time: 0.66 seconds

## Relational analysis of NS_A1_A1_A2_B1_A2_B2_B2

### Relational analysis result of NS_A1_A1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
time: 0.63 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -3027.0681152, 3314.9624023, -2837.3244629, 3110.5522461, -6137.6201172, 6152.2871094
1: -333.9471130, 230.2913361, -313.0335388, 215.5499878, -549.4970703, 543.3248901
2: -527.6524658, 620.9169922, -494.6603394, 581.9281006, -1109.5805664, 1115.5772705
3: -616.8242188, 389.8762207, -577.9241333, 365.5685120, -982.3927002, 967.8003540
4: -462.9597778, 502.2065125, -433.8983765, 471.2184448, -934.1781006, 936.1048584

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A2_B2_B1_B1_A1

### Relational analysis result of NS_A1_A1_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6861627, upper bound: 4810.6867046
time: 0.66 seconds

## Relational analysis of NS_A1_A1_A2_B2_B1_B1_A2

### Relational analysis result of NS_A1_A1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6861627, upper bound: 4810.6867046
time: 0.69 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -3112.5322266, 3410.6152344, -3140.7993164, 3440.4277344, -6552.9599609, 6551.4145508
1: -343.6705322, 237.0082397, -346.6357117, 239.3294067, -582.9999390, 583.6437378
2: -542.5199585, 639.6157837, -547.4165649, 645.5558472, -1188.0755615, 1187.0322266
3: -634.7329712, 401.1870117, -640.3034668, 404.7791748, -1039.5119629, 1041.4903564
4: -476.1308594, 516.9530029, -480.3262024, 521.7365112, -997.8673706, 997.2791748

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A2_B2_B1_B2_A1

### Relational analysis result of NS_A1_A1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6863452, upper bound: 4810.6873163
time: 0.74 seconds

## Relational analysis of NS_A1_A1_A2_B2_B1_B2_A2

### Relational analysis result of NS_A1_A1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6863452, upper bound: 4810.6873163
time: 0.69 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -3027.0681152, 3314.9624023, -2967.3193359, 3249.7133789, -6276.7812500, 6282.2812500
1: -333.9471130, 230.2913361, -326.9421997, 225.7568054, -559.7038574, 557.2335205
2: -527.6524658, 620.9169922, -516.8515625, 608.5002441, -1136.1527100, 1137.7684326
3: -616.8242188, 389.8762207, -603.0209351, 382.2441711, -999.0683594, 992.8971558
4: -462.9597778, 502.2065125, -452.7783508, 492.7452698, -955.7050781, 954.9848633

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A2_B2_B2_B1_A1

### Relational analysis result of NS_A1_A1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6812361, upper bound: 4810.6732626
time: 0.72 seconds

## Relational analysis of NS_A1_A1_A2_B2_B2_B1_A2

### Relational analysis result of NS_A1_A1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6812361, upper bound: 4810.6732626
time: 0.76 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -3112.5322266, 3410.6152344, -3319.5285645, 3628.5288086, -6741.0610352, 6730.1435547
1: -343.6705322, 237.0082397, -365.5800781, 253.3259735, -596.9964600, 602.5882568
2: -542.5199585, 639.6157837, -578.1290894, 681.4028320, -1223.9228516, 1217.7448730
3: -634.7329712, 401.1870117, -675.4342041, 427.3642273, -1062.0970459, 1076.6212158
4: -476.1308594, 516.9530029, -506.7695312, 550.6565552, -1026.7873535, 1023.7225342

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A2_B2_B2_B2_A1

### Relational analysis result of NS_A1_A1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6837092, upper bound: 4810.6764889
time: 0.73 seconds

## Relational analysis of NS_A1_A1_A2_B2_B2_B2_A2

### Relational analysis result of NS_A1_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6837092, upper bound: 4810.6764889
time: 0.90 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3038.7961426, 3324.9252930, -2028.9805908, 2286.5373535, -5325.3334961, 5353.9057617
1: -334.9035034, 231.5231628, -229.5176239, 153.8219452, -488.7254639, 461.0407715
2: -529.3840332, 623.9772339, -355.2395935, 430.0777588, -959.4617920, 979.2167969
3: -618.5256958, 391.2437744, -418.3032837, 267.5496216, -886.0752563, 809.5469971
4: -464.1603394, 504.5054016, -310.5647583, 347.5250549, -811.6854248, 815.0700073

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_A1_A2_B1_A2_B1

### Relational analysis result of NS_A1_A2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6664393, upper bound: 4810.6589634
time: 0.71 seconds

## Relational analysis of NS_A1_A2_A1_A2_B1_A2_B2

### Relational analysis result of NS_A1_A2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6664393, upper bound: 4810.6589634
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -3090.8012695, 3379.2165527, -3183.1872559, 3484.7131348, -6575.5146484, 6562.4018555
1: -340.4849854, 235.5458527, -351.1368713, 242.6456451, -583.1306152, 586.6825562
2: -538.4212036, 634.2402344, -554.7184448, 654.0677490, -1192.4890137, 1188.9584961
3: -629.1409302, 397.7811890, -648.7030640, 410.0805359, -1039.2214355, 1046.4842529
4: -472.1989746, 512.7055664, -486.6175537, 528.5173950, -1000.7163086, 999.3230591

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_A2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6922135, upper bound: 4810.6792465
time: 0.62 seconds

## Relational analysis of NS_A1_A2_A1_A2_B2_B1_B2

### Relational analysis result of NS_A1_A2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6950159, upper bound: 4810.6912221
time: 0.67 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3090.8012695, 3379.2165527, -3356.1091309, 3667.2805176, -6758.0820312, 6735.3247070
1: -340.4849854, 235.5458527, -369.5064392, 256.1694031, -596.6542358, 605.0523071
2: -538.4212036, 634.2402344, -584.4710693, 688.8901978, -1227.3114014, 1218.7113037
3: -629.1409302, 397.7811890, -682.7171021, 431.9926758, -1061.1335449, 1080.4982910
4: -472.1989746, 512.7055664, -512.2204590, 556.6329346, -1028.8317871, 1024.9260254

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_A2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6922135, upper bound: 4810.6792465
time: 0.87 seconds

## Relational analysis of NS_A1_A2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_A2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6950159, upper bound: 4810.6912221
time: 0.70 seconds

## BFS NS instance: NS_A1_A2_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -3076.9770508, 3376.3674316, -2047.8309326, 2306.4934082, -5383.4707031, 5424.1982422
1: -340.0244446, 234.4216766, -231.5381622, 155.2848358, -495.3092651, 465.9598389
2: -536.3371582, 633.4302368, -358.5087280, 433.8191833, -970.1563721, 991.9389038
3: -627.5025024, 396.9856567, -422.0451660, 269.9289551, -897.4313965, 819.0308228
4: -470.5467224, 511.9823608, -313.4083252, 350.5456848, -821.0924072, 825.3906860

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_A2_B1_A2_A1_B1

### Relational analysis result of NS_A1_A2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6521901, upper bound: 4810.6645738
time: 0.75 seconds

## Relational analysis of NS_A1_A2_A2_B1_A2_A1_B2

### Relational analysis result of NS_A1_A2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6521901, upper bound: 4810.6645738
time: 0.59 seconds

## BFS NS instance: NS_A1_A2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -3550.8383789, 3863.5771484, -2027.5803223, 2284.4042969, -5835.2426758, 5891.1572266
1: -391.6438599, 270.0361328, -229.2918243, 153.7321472, -545.3759155, 499.3279419
2: -619.3960571, 728.6742554, -354.9939270, 429.6048889, -1049.0007324, 1083.6682129
3: -730.2363892, 457.0833435, -417.9005737, 267.3137207, -997.5500488, 874.9838867
4: -546.9990845, 586.6268921, -310.3295288, 347.1771240, -894.1761475, 896.9563599

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_A2_B1_A2_A2_B1

### Relational analysis result of NS_A1_A2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6521901, upper bound: 4810.6645738
time: 0.65 seconds

## Relational analysis of NS_A1_A2_A2_B1_A2_A2_B2

### Relational analysis result of NS_A1_A2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6521901, upper bound: 4810.6645738
time: 0.70 seconds

## BFS NS instance: NS_A1_A2_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -1498.2170410, 1723.2938232, -3319.6479492, 3617.2741699, -5115.4897461, 5042.9418945
1: -172.1239166, 112.7873917, -364.4071655, 253.2551575, -425.3790894, 477.1945190
2: -263.1828918, 321.6305847, -577.9390259, 678.7923584, -941.9752197, 899.5695801
3: -311.1298523, 200.0741577, -673.6901245, 426.2806091, -737.4104614, 873.7642212
4: -230.4317780, 260.8777771, -506.2711182, 548.8581543, -779.2899170, 767.1489258

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_A2_B2_A1_A1_B1

### Relational analysis result of NS_A1_A2_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6647123, upper bound: 4810.6612757
time: 0.78 seconds

## Relational analysis of NS_A1_A2_A2_B2_A1_A1_B2

### Relational analysis result of NS_A1_A2_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6617361, upper bound: 4810.6503313
time: 0.63 seconds

## BFS NS instance: NS_A1_A2_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -1765.6187744, 2015.7495117, -3417.3586426, 3726.3222656, -5491.9404297, 5433.1083984
1: -201.9225464, 133.5314331, -375.4983521, 260.9391785, -462.8617249, 509.0297852
2: -309.6832275, 378.1722412, -594.9777222, 700.0219116, -1009.7050171, 973.1499634
3: -365.8813782, 234.8818665, -694.2208252, 439.1720276, -805.0532837, 929.1026611
4: -271.0461121, 305.8213501, -521.3200684, 565.6181030, -836.6641846, 827.1414185

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_A2_B2_A1_A2_B1

### Relational analysis result of NS_A1_A2_A2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6494992, upper bound: 4810.6527379
time: 0.75 seconds

## Relational analysis of NS_A1_A2_A2_B2_A1_A2_B2

### Relational analysis result of NS_A1_A2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6736946, upper bound: 4810.6652654
time: 0.73 seconds

## BFS NS instance: NS_A1_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3133.2282715, 3434.1193848, -3188.2343750, 3489.7519531, -6622.9799805, 6622.3530273
1: -345.9767761, 238.7386017, -351.6493530, 243.0374298, -589.0141602, 590.3879395
2: -546.1246338, 644.4014893, -555.5917358, 655.0114746, -1201.1361084, 1199.9931641
3: -638.9020996, 403.9585571, -649.6787720, 410.6920776, -1049.5941162, 1053.6373291
4: -479.1801758, 520.7603149, -487.3728638, 529.2781372, -1008.4583130, 1008.1331177

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_A2_B2_A2_B1_B1

### Relational analysis result of NS_A1_A2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6945674, upper bound: 4810.6827512
time: 0.74 seconds

## Relational analysis of NS_A1_A2_A2_B2_A2_B1_B2

### Relational analysis result of NS_A1_A2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6952668, upper bound: 4810.6918457
time: 0.67 seconds

## BFS NS instance: NS_A1_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3133.2282715, 3434.1193848, -3360.9421387, 3672.1091309, -6805.3359375, 6795.0610352
1: -345.9767761, 238.7386017, -369.9964905, 256.5442200, -602.5209351, 608.7351074
2: -546.1246338, 644.4014893, -585.3003540, 689.7937012, -1235.9183350, 1229.7014160
3: -638.9020996, 403.9585571, -683.6477661, 432.5768127, -1071.4786377, 1087.6062012
4: -479.1801758, 520.7603149, -512.9323120, 557.3615723, -1036.5417480, 1033.6926270

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A1_A2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6945674, upper bound: 4810.6827512
time: 0.69 seconds

## Relational analysis of NS_A1_A2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A1_A2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6952668, upper bound: 4810.6918457
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -2146.4965820, 2358.7976074, -2063.5051270, 2266.3430176, -4412.8398438, 4422.3022461
1: -237.0368805, 163.1379242, -227.6954803, 156.8362885, -393.8731079, 390.8334045
2: -374.3010254, 440.9637146, -359.4161072, 423.7663269, -798.0673828, 800.3796387
3: -435.5907593, 276.5790405, -418.4036560, 265.5219421, -701.1126709, 694.9826050
4: -326.7074890, 357.2184143, -313.7945251, 343.1575317, -669.8649902, 671.0128784

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6618342, upper bound: 4810.6559144
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6618342, upper bound: 4810.6586191
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -2456.6293945, 2693.8178711, -2063.5051270, 2266.3430176, -4722.9721680, 4757.3232422
1: -271.1279602, 187.1837769, -227.6954803, 156.8362885, -427.9642029, 414.8792419
2: -428.1947937, 504.9926147, -359.4161072, 423.7663269, -851.9611206, 864.4086304
3: -498.7737427, 316.4996948, -418.4036560, 265.5219421, -764.2956543, 734.9033203
4: -373.9003906, 408.3195801, -313.7945251, 343.1575317, -717.0578613, 722.1141357

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6618342, upper bound: 4810.6559144
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6618342, upper bound: 4810.6586191
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -2146.4965820, 2358.7976074, -2379.4299316, 2609.7792969, -4756.2758789, 4738.2275391
1: -237.0368805, 163.1379242, -262.6293945, 181.3446045, -418.3814392, 425.7673340
2: -374.3010254, 440.9637146, -414.5296021, 489.5097961, -863.8107910, 855.4932251
3: -435.5907593, 276.5790405, -483.2355042, 306.4008789, -741.9916382, 759.8144531
4: -326.7074890, 357.2184143, -362.1143799, 395.6056824, -722.3131714, 719.3327637

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6583292, upper bound: 4810.6549312
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6583292, upper bound: 4810.6586677
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -2456.6293945, 2693.8178711, -2379.4299316, 2609.7792969, -5066.4086914, 5073.2480469
1: -271.1279602, 187.1837769, -262.6293945, 181.3446045, -452.4725037, 449.8131714
2: -428.1947937, 504.9926147, -414.5296021, 489.5097961, -917.7045898, 919.5222168
3: -498.7737427, 316.4996948, -483.2355042, 306.4008789, -805.1745605, 799.7352295
4: -373.9003906, 408.3195801, -362.1143799, 395.6056824, -769.5061035, 770.4339600

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6583292, upper bound: 4810.6549312
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6583292, upper bound: 4810.6586191
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -2374.2180176, 2600.9721680, -2184.3613281, 2397.4636230, -4771.6801758, 4785.3334961
1: -261.7128296, 180.8109131, -240.9242096, 166.1634979, -427.8763428, 421.7350464
2: -413.6471863, 487.2210388, -380.5613098, 448.6642456, -862.3113403, 867.7823486
3: -481.5046387, 305.5551453, -442.7240295, 281.2422791, -762.7469482, 748.2791748
4: -361.0887756, 394.1454773, -332.0378113, 363.3406982, -724.4294434, 726.1832886

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6573729, upper bound: 4810.6555752
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6573729, upper bound: 4810.6555752
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -2146.4965820, 2358.7976074, -2552.9064941, 2795.0869141, -4941.5834961, 4911.7041016
1: -237.0368805, 163.1379242, -281.4103699, 194.7635498, -431.8003845, 444.5482788
2: -374.3010254, 440.9637146, -444.7479248, 524.6442871, -898.9453125, 885.7114258
3: -435.5907593, 276.5790405, -518.0458984, 328.6477051, -764.2384644, 794.6248779
4: -326.7074890, 357.2184143, -388.3470154, 424.0177917, -750.7252808, 745.5653687

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6453548, upper bound: 4810.6453548
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6453548, upper bound: 4810.6586677
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -2456.6293945, 2693.8178711, -2552.9064941, 2795.0869141, -5251.7163086, 5246.7246094
1: -271.1279602, 187.1837769, -281.4103699, 194.7635498, -465.8914490, 468.5941467
2: -428.1947937, 504.9926147, -444.7479248, 524.6442871, -952.8391113, 949.7404785
3: -498.7737427, 316.4996948, -518.0458984, 328.6477051, -827.4214478, 834.5455933
4: -373.9003906, 408.3195801, -388.3470154, 424.0177917, -797.9182129, 796.6666260

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6453548, upper bound: 4810.6453548
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6453548, upper bound: 4810.6586191
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -3276.8461914, 3573.7128906, -2052.8195801, 2311.5190430, -5588.3647461, 5626.5322266
1: -360.0913696, 250.0688324, -232.0591431, 155.6458435, -515.7371826, 482.1279297
2: -570.4400635, 671.3382568, -359.3649292, 434.8150024, -1005.2549438, 1030.7031250
3: -665.5194092, 421.1709290, -423.0617371, 270.5424194, -936.0617676, 844.2326660
4: -499.6944885, 542.6590576, -314.1549683, 351.3499146, -851.0443726, 856.8140259

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6501457, upper bound: 4810.6341626
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_A1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6551754, upper bound: 4810.6415803
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3278.8442383, 3575.5529785, -3423.8503418, 3732.9907227, -7011.8330078, 6999.4018555
1: -360.2859497, 250.2220306, -376.1878662, 261.4407349, -621.7266846, 626.4099121
2: -570.7663574, 671.6951294, -596.0693359, 701.3428345, -1272.1091309, 1267.7642822
3: -665.8966064, 421.4064636, -695.5516357, 439.9843445, -1105.8809814, 1116.9580078
4: -499.9883728, 542.9427490, -522.2832031, 566.6658325, -1066.6541748, 1065.2255859

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6611105, upper bound: 4810.6889653
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_A1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6611105, upper bound: 4810.6889653
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1962.4527588, 2219.7587891, -2120.6821289, 2367.4853516, -4329.9379883, 4340.4409180
1: -222.7405701, 148.6508331, -237.8793030, 161.1598816, -383.9004211, 386.5301514
2: -343.7460022, 417.3294373, -371.3251343, 444.3106079, -788.0566406, 788.6544800
3: -405.2289429, 259.5137939, -435.6360474, 276.8802490, -682.1088867, 695.1497192
4: -300.6129761, 337.2887268, -324.9262695, 359.0278931, -659.6408081, 662.2149048

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6495243, upper bound: 4810.6539321
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6495243, upper bound: 4810.6709238
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3307.9858398, 3618.2675781, -2351.7456055, 2579.6193848, -5887.6049805, 5970.0126953
1: -364.5164490, 252.4143982, -259.4494934, 179.3026886, -543.8191528, 511.8638916
2: -576.2164917, 679.5309448, -409.0761108, 484.2343445, -1060.4508057, 1088.6068115
3: -673.2866211, 426.0801697, -477.1190491, 302.5578613, -975.8444824, 903.1990967
4: -505.0744629, 549.1187134, -357.4937439, 391.0996399, -896.1740723, 906.6124268

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6680596, upper bound: 4810.6771352
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6680596, upper bound: 4810.6965772
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1962.4527588, 2219.7587891, -2297.7019043, 2553.8076172, -4516.2602539, 4517.4609375
1: -222.7405701, 148.6508331, -256.7595520, 174.9562988, -397.6968384, 405.4103699
2: -343.7460022, 417.3294373, -401.7710571, 479.8776855, -823.6236572, 819.1004639
3: -405.2289429, 259.5137939, -470.9323425, 299.3013000, -704.5302734, 730.4461670
4: -300.6129761, 337.2887268, -351.3162231, 387.6761780, -688.2890625, 688.6049805

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6454544, upper bound: 4810.6454544
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6454544, upper bound: 4810.6633988
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3307.9858398, 3618.2675781, -2520.5798340, 2763.0266113, -6071.0126953, 6138.8476562
1: -364.5164490, 252.4143982, -277.9851685, 192.5141907, -557.0305786, 530.3995361
2: -576.2164917, 679.5309448, -437.9286499, 519.0756836, -1095.2919922, 1117.4594727
3: -673.2866211, 426.0801697, -509.9381409, 324.6102600, -997.8968506, 936.0182495
4: -505.0744629, 549.1187134, -382.1120605, 419.2762146, -924.3507080, 931.2307739

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6633988, upper bound: 4810.6688223
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6633988, upper bound: 4810.6905492
time: 0.68 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.61 seconds
NS_A1_A1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6605612, upper bound: 4810.6510551
NS_A1_A1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6605612, upper bound: 4810.6510653
NS_A1_A1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6605612, upper bound: 4810.6510551
NS_A1_A1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6605612, upper bound: 4810.6510653
NS_A1_A1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6613534, upper bound: 4810.6515113
NS_A1_A1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6613534, upper bound: 4810.6515112
NS_A1_A1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6614472, upper bound: 4810.6627782
NS_A1_A1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6614472, upper bound: 4810.6627782
NS_A1_A1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6614472, upper bound: 4810.6627782
NS_A1_A1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6614472, upper bound: 4810.6627782
NS_A1_A1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
NS_A1_A1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
NS_A1_A1_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
NS_A1_A1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
NS_A1_A1_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6861627, upper bound: 4810.6867046
NS_A1_A1_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6861627, upper bound: 4810.6867046
NS_A1_A1_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6863452, upper bound: 4810.6873163
NS_A1_A1_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6863452, upper bound: 4810.6873163
NS_A1_A1_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6812361, upper bound: 4810.6732626
NS_A1_A1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6812361, upper bound: 4810.6732626
NS_A1_A1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6837092, upper bound: 4810.6764889
NS_A1_A1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6837092, upper bound: 4810.6764889
NS_A1_A2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6664393, upper bound: 4810.6589634
NS_A1_A2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6664393, upper bound: 4810.6589634
NS_A1_A2_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6922135, upper bound: 4810.6792465
NS_A1_A2_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6950159, upper bound: 4810.6912221
NS_A1_A2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6922135, upper bound: 4810.6792465
NS_A1_A2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6950159, upper bound: 4810.6912221
NS_A1_A2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6521901, upper bound: 4810.6645738
NS_A1_A2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6521901, upper bound: 4810.6645738
NS_A1_A2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6521901, upper bound: 4810.6645738
NS_A1_A2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6521901, upper bound: 4810.6645738
NS_A1_A2_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6647123, upper bound: 4810.6612757
NS_A1_A2_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6617361, upper bound: 4810.6503313
NS_A1_A2_A2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6494992, upper bound: 4810.6527379
NS_A1_A2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6736946, upper bound: 4810.6652654
NS_A1_A2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6945674, upper bound: 4810.6827512
NS_A1_A2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6952668, upper bound: 4810.6918457
NS_A1_A2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6945674, upper bound: 4810.6827512
NS_A1_A2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6952668, upper bound: 4810.6918457
NS_A2_B2_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6618342, upper bound: 4810.6559144
NS_A2_B2_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6618342, upper bound: 4810.6586191
NS_A2_B2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6618342, upper bound: 4810.6559144
NS_A2_B2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6618342, upper bound: 4810.6586191
NS_A2_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6583292, upper bound: 4810.6549312
NS_A2_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6583292, upper bound: 4810.6586677
NS_A2_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6583292, upper bound: 4810.6549312
NS_A2_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6583292, upper bound: 4810.6586191
NS_A2_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6573729, upper bound: 4810.6555752
NS_A2_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6573729, upper bound: 4810.6555752
NS_A2_B2_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6453548, upper bound: 4810.6453548
NS_A2_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6453548, upper bound: 4810.6586677
NS_A2_B2_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6453548, upper bound: 4810.6453548
NS_A2_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6453548, upper bound: 4810.6586191
NS_A2_B2_A2_A1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6501457, upper bound: 4810.6341626
NS_A2_B2_A2_A1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6551754, upper bound: 4810.6415803
NS_A2_B2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6611105, upper bound: 4810.6889653
NS_A2_B2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6611105, upper bound: 4810.6889653
NS_A2_B2_A2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6495243, upper bound: 4810.6539321
NS_A2_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6495243, upper bound: 4810.6709238
NS_A2_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6680596, upper bound: 4810.6771352
NS_A2_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6680596, upper bound: 4810.6965772
NS_A2_B2_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6454544, upper bound: 4810.6454544
NS_A2_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6454544, upper bound: 4810.6633988
NS_A2_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6633988, upper bound: 4810.6688223
NS_A2_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 0, lower bound: -4810.6633988, upper bound: 4810.6905492

## BFS NS instance: NS_A1_A1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1478.4554443, 1702.3707275, -3254.5468750, 3551.0053711, -5029.4604492, 4956.9165039
1: -170.0764313, 111.1669083, -357.6096802, 248.2638245, -418.3402710, 468.7765808
2: -259.9661255, 317.3677063, -566.7358398, 666.2329102, -926.1989746, 884.1035156
3: -307.5666809, 197.5804749, -660.8551636, 418.2845154, -725.8511963, 858.4356689
4: -227.8266144, 257.4858093, -496.4418640, 538.7809448, -766.6075439, 753.9276733

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_A1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6632793, upper bound: 4810.6597850
time: 0.65 seconds

## Relational analysis of NS_A1_A1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_A1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6632792, upper bound: 4810.6597850
time: 0.66 seconds

## BFS NS instance: NS_A1_A1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2042.3602295, 2292.7126465, -3254.6638184, 3551.1137695, -5593.4731445, 5547.3764648
1: -231.5921021, 153.6623230, -357.6212158, 248.2728424, -479.8649292, 511.2835388
2: -358.7408447, 430.8187866, -566.7550049, 666.2540894, -1024.9948730, 997.5737305
3: -427.2606201, 269.1007385, -660.8775024, 418.2984314, -845.5588379, 929.9782715
4: -317.0040588, 347.2101746, -496.4588928, 538.7975464, -855.8016357, 843.6690674

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_A1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6632792, upper bound: 4810.6597850
time: 0.70 seconds

## Relational analysis of NS_A1_A1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_A1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6632792, upper bound: 4810.6597850
time: 0.72 seconds

## BFS NS instance: NS_A1_A1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1478.4554443, 1702.3707275, -3638.2517090, 3941.7033691, -5420.1586914, 5340.6225586
1: -170.0764313, 111.1669083, -399.4878845, 276.9766846, -447.0531006, 510.4671326
2: -259.9661255, 317.3677063, -634.2360229, 743.2808228, -1003.2468872, 951.6036987
3: -307.5666809, 197.5804749, -745.5236816, 466.8120728, -774.3787842, 943.1041260
4: -227.8266144, 257.4858093, -559.0594482, 598.7807007, -826.6072998, 816.5452271

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_A1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6605612, upper bound: 4810.6510551
time: 0.64 seconds

## Relational analysis of NS_A1_A1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_A1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6605612, upper bound: 4810.6510551
time: 0.80 seconds

## BFS NS instance: NS_A1_A1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2042.3602295, 2292.7126465, -3638.3686523, 3941.8125000, -5974.4101562, 5931.0810547
1: -231.5921021, 153.6623230, -399.4993286, 276.9856873, -508.5777588, 552.1706543
2: -358.7408447, 430.8187866, -634.2553101, 743.3018799, -1100.0722656, 1065.0737305
3: -427.2606201, 269.1007385, -745.5458984, 466.8259583, -892.6809082, 1014.6393433
4: -317.0040588, 347.2101746, -559.0765991, 598.7974243, -914.2893066, 906.2867432

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_A1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6605612, upper bound: 4810.6510653
time: 0.68 seconds

## Relational analysis of NS_A1_A1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_A1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6605612, upper bound: 4810.6510653
time: 0.83 seconds

## BFS NS instance: NS_A1_A1_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1724.0186768, 1971.2467041, -2961.6638184, 3240.3833008, -4963.2504883, 4932.9101562
1: -197.4126282, 130.2012329, -325.9452820, 225.4133301, -422.8258972, 456.1464539
2: -302.5283508, 369.2284546, -515.7872314, 606.8123779, -909.3406982, 885.0155640
3: -357.5392151, 229.5215454, -601.3654785, 381.1320190, -738.6712036, 830.8870239
4: -264.8898315, 298.7161865, -451.7415161, 491.3718567, -756.2617188, 750.4575806

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A1_B2_A2_B2_B1_B1

### Relational analysis result of NS_A1_A1_A1_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6613534, upper bound: 4810.6515113
time: 0.63 seconds

## Relational analysis of NS_A1_A1_A1_B2_A2_B2_B1_B2

### Relational analysis result of NS_A1_A1_A1_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6613534, upper bound: 4810.6515113
time: 0.66 seconds

## BFS NS instance: NS_A1_A1_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -1724.0186768, 1971.2467041, -3299.2485352, 3604.1564941, -5328.1752930, 5270.4951172
1: -197.4126282, 130.2012329, -363.0193787, 251.7882996, -449.2008667, 493.2205811
2: -302.5283508, 369.2284546, -574.5494385, 676.8510132, -979.3792725, 943.7777710
3: -357.5392151, 229.5215454, -670.7481079, 424.4927979, -782.0319824, 900.2696533
4: -264.8898315, 298.7161865, -503.3264160, 547.0652466, -811.9550781, 802.0424805

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A1_B2_A2_B2_B2_B1

### Relational analysis result of NS_A1_A1_A1_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6613534, upper bound: 4810.6515113
time: 0.67 seconds

## Relational analysis of NS_A1_A1_A1_B2_A2_B2_B2_B2

### Relational analysis result of NS_A1_A1_A1_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6613534, upper bound: 4810.6515113
time: 0.66 seconds

## BFS NS instance: NS_A1_A1_A2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -3054.5959473, 3350.6298828, -1801.4993896, 2053.5119629, -5108.1079102, 5152.1289062
1: -337.4991150, 232.5513153, -205.7566681, 136.3460846, -473.8451843, 438.3079224
2: -532.4488525, 628.2415771, -315.8412170, 385.5733032, -918.0220947, 944.0826416
3: -622.9971924, 393.9600525, -373.0405884, 239.4371338, -862.4342041, 767.0005493
4: -467.2411804, 507.8461914, -276.4072571, 311.7078552, -778.9490356, 784.2534180

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A2_B1_A1_B1_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6595195, upper bound: 4810.6604380
time: 0.70 seconds

## Relational analysis of NS_A1_A1_A2_B1_A1_B1_B1_B2

### Relational analysis result of NS_A1_A1_A2_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6594077, upper bound: 4810.6606189
time: 0.66 seconds

## BFS NS instance: NS_A1_A1_A2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -3054.6242676, 3350.6567383, -1967.1918945, 2224.3715820, -5278.9960938, 5317.8486328
1: -337.5019226, 232.5534973, -223.1809692, 149.0669861, -486.5689087, 455.7344055
2: -532.4536133, 628.2467651, -344.4151001, 418.3011475, -950.7547607, 972.6618652
3: -623.0026245, 393.9634094, -405.8856812, 260.0877686, -883.0903931, 799.8491211
4: -467.2454224, 507.8503723, -301.1433411, 338.0727844, -805.3182373, 808.9937134

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A2_B1_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_A2_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6595195, upper bound: 4810.6604380
time: 0.63 seconds

## Relational analysis of NS_A1_A1_A2_B1_A1_B1_B2_B2

### Relational analysis result of NS_A1_A1_A2_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6594077, upper bound: 4810.6606189
time: 0.68 seconds

## BFS NS instance: NS_A1_A1_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -3054.7668457, 3350.7915039, -2279.3664551, 2554.8750000, -5609.6411133, 5630.1567383
1: -337.5160217, 232.5644684, -258.1563721, 172.4020081, -509.9180298, 490.7208252
2: -532.4766846, 628.2727051, -399.5362549, 482.6531982, -1015.1298828, 1027.8089600
3: -623.0292358, 393.9805298, -475.6488953, 300.3447266, -923.3739014, 869.6293335
4: -467.2661133, 507.8709717, -352.4555359, 388.0728760, -855.3389282, 860.3264771

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A2_B1_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6429431, upper bound: 4810.6372268
time: 0.68 seconds

## Relational analysis of NS_A1_A1_A2_B1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_A2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_A2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_A2_B1_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6564556, upper bound: 4810.6550696
time: 0.82 seconds

## Relational analysis of NS_A1_A1_A2_B1_A1_B2_B1_B2

### Relational analysis result of NS_A1_A1_A2_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6573746, upper bound: 4810.6585096
time: 0.72 seconds

## BFS NS instance: NS_A1_A1_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -3054.7727051, 3350.7966309, -2325.7316895, 2595.7526855, -5650.5244141, 5676.5283203
1: -337.5166321, 232.5649109, -262.3696899, 176.1119690, -513.6286011, 494.9345703
2: -532.4776611, 628.2737427, -407.4614563, 490.9046326, -1023.3822632, 1035.7352295
3: -623.0303955, 393.9812317, -484.2727661, 305.5856934, -928.6160889, 878.2540283
4: -467.2669373, 507.8717957, -359.0940247, 394.7297363, -861.9966431, 866.9658203

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A2_B1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_A2_B1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6429431, upper bound: 4810.6441491
time: 0.66 seconds

## Relational analysis of NS_A1_A1_A2_B1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_A2_B1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_A2_B1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6564556, upper bound: 4810.6550696
time: 0.65 seconds

## Relational analysis of NS_A1_A1_A2_B1_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_A2_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6573746, upper bound: 4810.6585096
time: 0.70 seconds

## BFS NS instance: NS_A1_A1_A2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3594.0441895, 3908.2885742, -1801.4993896, 2053.5119629, -5647.5561523, 5709.7880859
1: -396.2311707, 273.2227783, -205.7566681, 136.3460846, -532.5772095, 478.9794006
2: -626.8261719, 736.7796021, -315.8412170, 385.5733032, -1012.3992920, 1052.6208496
3: -738.7535400, 462.4617310, -373.0405884, 239.4371338, -978.1906128, 835.5022583
4: -553.5611572, 593.1870117, -276.4072571, 311.7078552, -865.2689819, 869.5942383

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_A2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_A2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_A2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A2_B1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6502082, upper bound: 4810.6670954
time: 0.61 seconds

## Relational analysis of NS_A1_A1_A2_B1_A2_B1_B1_B2

### Relational analysis result of NS_A1_A1_A2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6498213, upper bound: 4810.6624930
time: 0.69 seconds

## BFS NS instance: NS_A1_A1_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3594.0749512, 3908.3171387, -1967.1918945, 2224.3715820, -5818.4453125, 5875.5087891
1: -396.2342224, 273.2251892, -223.1809692, 149.0669861, -545.3009644, 496.4060669
2: -626.8310547, 736.7851562, -344.4151001, 418.3011475, -1045.1322021, 1081.2001953
3: -738.7592773, 462.4653320, -405.8856812, 260.0877686, -998.8470459, 868.3510132
4: -553.5656128, 593.1913452, -301.1433411, 338.0727844, -891.6384277, 894.3345947

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A2_B1_A2_B1_B2_B1

### Relational analysis result of NS_A1_A1_A2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6491638, upper bound: 4810.6628179
time: 0.60 seconds

## Relational analysis of NS_A1_A1_A2_B1_A2_B1_B2_B2

### Relational analysis result of NS_A1_A1_A2_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6502082, upper bound: 4810.6670954
time: 0.65 seconds

## BFS NS instance: NS_A1_A1_A2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -3594.2751465, 3908.5053711, -2279.3664551, 2554.8750000, -6149.1494141, 6187.2304688
1: -396.2538147, 273.2405090, -258.1563721, 172.4020081, -568.3942261, 531.3968506
2: -626.8639526, 736.8209229, -399.5362549, 482.6531982, -1109.5170898, 1136.2521973
3: -738.7969971, 462.4891968, -475.6488953, 300.3447266, -1039.1417236, 938.1358643
4: -553.5949707, 593.2200317, -352.4555359, 388.0728760, -941.6677246, 945.6320801

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_A2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_A2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A2_B1_A2_B2_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6501477, upper bound: 4810.6627782
time: 0.58 seconds

## Relational analysis of NS_A1_A1_A2_B1_A2_B2_B1_B2

### Relational analysis result of NS_A1_A1_A2_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6498213, upper bound: 4810.6620535
time: 0.69 seconds

## BFS NS instance: NS_A1_A1_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3594.2810059, 3908.5109863, -2325.7316895, 2595.7526855, -6190.0327148, 6234.2426758
1: -396.2543945, 273.2409363, -262.3696899, 176.1119690, -572.2225952, 535.6105957
2: -626.8649902, 736.8220825, -407.4614563, 490.9046326, -1117.7694092, 1144.2834473
3: -738.7980347, 462.4899292, -484.2727661, 305.5856934, -1044.3835449, 946.7625732
4: -553.5958862, 593.2209473, -359.0940247, 394.7297363, -948.3256226, 952.3149414

Time for backsubstitution: 2.81 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.70 + 416.50 = 421.20 seconds
