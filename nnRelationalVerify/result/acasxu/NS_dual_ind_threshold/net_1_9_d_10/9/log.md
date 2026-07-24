## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 2.87951805


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807)
1: (-14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082)
2: (-7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548)
3: (-9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106)
4: (-5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.23 + 1.48 = 2.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3.1994645, upper bound: 3.1994645

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1337461, upper bound: 3.1297437
time: 0.46 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1026031, upper bound: 3.1026031
time: 0.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.10 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 0, lower bound: -3.1337461, upper bound: 3.1297437
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 0, lower bound: -3.1026031, upper bound: 3.1026031

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1.8310382, 1.9572052, -1.9336649, 2.0393150, -3.8703530, 3.8908701
1: -13.5492735, 4.3171434, -13.9637461, 4.5341206, -18.0833931, 18.2808876
2: -6.9885325, 4.4688768, -7.3317533, 4.6654143, -11.6539450, 11.8006277
3: -9.0682240, 3.2941692, -9.4134150, 3.4265895, -12.4948139, 12.7075815
4: -4.8162956, 3.7064695, -5.0821681, 3.8676977, -8.6839933, 8.7886372

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1026031, upper bound: 3.1026031
time: 0.65 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1026031, upper bound: 3.1026031
time: 0.56 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2.1510289, 2.2559071, -1.7830794, 1.9222810, -4.0733099, 4.0389857
1: -14.7173424, 4.9412661, -13.2892399, 4.2039771, -18.9213200, 18.2305069
2: -7.8387775, 5.0611115, -6.7991495, 4.3732357, -12.2120132, 11.8602600
3: -9.9205475, 3.7001772, -8.8670549, 3.2182777, -13.1388254, 12.5672312
4: -5.5745769, 4.2633672, -4.6842375, 3.6582992, -9.2328758, 8.9476042

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0738943, upper bound: 3.0745898
time: 0.50 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0647342, upper bound: 3.0647342
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.31 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -3.1026031, upper bound: 3.1026031
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -3.1026031, upper bound: 3.1026031
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -3.0738943, upper bound: 3.0745898
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -3.0647342, upper bound: 3.0647342

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -1.8310382, 1.9572052, -1.8310382, 1.9572052, -3.7882433, 3.7882433
1: -13.5492735, 4.3171434, -13.5492735, 4.3171434, -17.8664169, 17.8664169
2: -6.9885325, 4.4688768, -6.9885325, 4.4688768, -11.4574070, 11.4574070
3: -9.0682240, 3.2941692, -9.0682240, 3.2941692, -12.3623905, 12.3623905
4: -4.8162956, 3.7064695, -4.8162956, 3.7064695, -8.5227633, 8.5227642

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0982168, upper bound: 3.0943154
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1069409, upper bound: 3.1018577
time: 0.54 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1.8310382, 1.9572052, -2.1510289, 2.2559071, -4.0869436, 4.1082339
1: -13.5492735, 4.3171434, -14.7173424, 4.9412661, -18.4905396, 19.0344849
2: -6.9885325, 4.4688768, -7.8387775, 5.0611115, -12.0496426, 12.3076525
3: -9.0682240, 3.2941692, -9.9205475, 3.7001772, -12.7684011, 13.2147160
4: -4.8162956, 3.7064695, -5.5745769, 4.2633672, -9.0796623, 9.2810450

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0982168, upper bound: 3.0943154
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1069409, upper bound: 3.1018577
time: 0.54 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1.9349504, 2.0685804, -1.1641695, 1.4680066, -3.4029570, 3.2327497
1: -13.8081121, 4.4766283, -11.1120930, 2.8522198, -16.6603317, 15.5887213
2: -7.1310072, 4.6415892, -4.7768183, 3.2429109, -10.3739176, 9.4184074
3: -9.2060404, 3.4046984, -6.9579725, 2.4041092, -11.6101494, 10.3626690
4: -5.0181427, 3.9166758, -3.1489544, 2.8241675, -7.8423100, 7.0656300

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0647342, upper bound: 3.0647342
time: 0.51 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0647342, upper bound: 3.0647342
time: 0.55 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -2.0516081, 2.1700487, -1.3304267, 1.5554013, -3.6070094, 3.5004754
1: -14.3025160, 4.7339621, -11.2004385, 3.2161264, -17.5186424, 15.9344006
2: -7.5245175, 4.8717394, -5.2691312, 3.4713333, -10.9958506, 10.1408672
3: -9.5954552, 3.5672021, -7.2221880, 2.5724189, -12.1678724, 10.7893877
4: -5.3181911, 4.1022468, -3.5299795, 2.9892185, -8.3074093, 7.6322255

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0647342, upper bound: 3.0647342
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0647342, upper bound: 3.0647342
time: 0.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.32 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -3.0982168, upper bound: 3.0943154
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -3.1069409, upper bound: 3.1018577
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -3.0982168, upper bound: 3.0943154
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -3.1069409, upper bound: 3.1018577
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -3.0647342, upper bound: 3.0647342
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -3.0647342, upper bound: 3.0647342
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -3.0647342, upper bound: 3.0647342
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -3.0647342, upper bound: 3.0647342

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.2097423, 1.4932493, -1.6064715, 1.7701339, -2.9798758, 3.0997207
1: -11.2694292, 2.9661100, -12.5582399, 3.8286660, -15.0980949, 15.5243473
2: -4.9715719, 3.3173807, -6.2409816, 4.0279436, -8.9995136, 9.5583620
3: -7.1621819, 2.4731677, -8.2970791, 2.9783316, -10.1405125, 10.7702465
4: -3.2760181, 2.8666625, -4.2472639, 3.3685188, -6.6445370, 7.1139259

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0773033, upper bound: 3.0834000
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0513261, upper bound: 3.0530705
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.3914477, 1.5896201, -1.7197154, 1.8596777, -3.2511244, 3.3093355
1: -11.4863129, 3.3692830, -13.0526552, 4.0794153, -15.5657282, 16.4219379
2: -5.5162153, 3.5947320, -6.6215053, 4.2493191, -9.7655334, 10.2162371
3: -7.4902196, 2.6684668, -8.6849003, 3.1385236, -10.6287432, 11.3533669
4: -3.6932654, 3.0478096, -4.5342131, 3.5344760, -7.2277412, 7.5820227

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1297552, upper bound: 3.1359257
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0946588, upper bound: 3.0946588
time: 0.43 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.2097423, 1.4932493, -1.9349504, 2.0685804, -3.2783222, 3.4281998
1: -11.2694292, 2.9661100, -13.8081121, 4.4766283, -15.7460575, 16.7742214
2: -4.9715719, 3.3173807, -7.1310072, 4.6415892, -9.6131611, 10.4483881
3: -7.1621819, 2.4731677, -9.2060404, 3.4046984, -10.5668783, 11.6792078
4: -3.2760181, 2.8666625, -5.0181427, 3.9166758, -7.1926932, 7.8848038

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0982168, upper bound: 3.0943154
time: 0.48 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0982168, upper bound: 3.0943154
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.3914477, 1.5896201, -2.0516081, 2.1700487, -3.5614963, 3.6412282
1: -11.4863129, 3.3692830, -14.3025160, 4.7339621, -16.2202759, 17.6717987
2: -5.5162153, 3.5947320, -7.5245175, 4.8717394, -10.3879538, 11.1192493
3: -7.4902196, 2.6684668, -9.5954552, 3.5672021, -11.0574207, 12.2639208
4: -3.6932654, 3.0478096, -5.3181911, 4.1022468, -7.7955103, 8.3660011

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1069409, upper bound: 3.1018577
time: 0.40 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1069409, upper bound: 3.1018577
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.5155612, 1.7437913, -1.1641695, 1.4680066, -2.9835677, 2.9079609
1: -12.4957151, 3.5932887, -11.1120930, 2.8522198, -15.3479347, 14.7053814
2: -5.8347397, 3.8786392, -4.7768183, 3.2429109, -9.0776491, 8.6554575
3: -8.0386419, 2.8724000, -6.9579725, 2.4041092, -10.4427500, 9.8303719
4: -3.9789469, 3.3175647, -3.1489544, 2.8241675, -6.8031144, 6.4665189

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0482081, upper bound: 3.0514635
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.7273169, 1.8819833, -1.1641695, 1.4680066, -3.1953235, 3.0461521
1: -12.8924522, 4.0498548, -11.1120930, 2.8522198, -15.7446718, 15.1619473
2: -6.4719362, 4.2366500, -4.7768183, 3.2429109, -9.7148466, 9.0134678
3: -8.5064764, 3.1202321, -6.9579725, 2.4041092, -10.9105854, 10.0782032
4: -4.4789262, 3.5923789, -3.1489544, 2.8241675, -7.3030939, 6.7413330

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0482081, upper bound: 3.0514635
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.5155612, 1.7437913, -1.3304267, 1.5554013, -3.0709624, 3.0742180
1: -12.4957151, 3.5932887, -11.2004385, 3.2161264, -15.7118416, 14.7937260
2: -5.8347397, 3.8786392, -5.2691312, 3.4713333, -9.3060703, 9.1477690
3: -8.0386419, 2.8724000, -7.2221880, 2.5724189, -10.6110611, 10.0945873
4: -3.9789469, 3.3175647, -3.5299795, 2.9892185, -6.9681654, 6.8475432

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0346686, upper bound: 3.0359654
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.7273169, 1.8819833, -1.3304267, 1.5554013, -3.2827182, 3.2124095
1: -12.8924522, 4.0498548, -11.2004385, 3.2161264, -16.1085777, 15.2502899
2: -6.4719362, 4.2366500, -5.2691312, 3.4713333, -9.9432697, 9.5057783
3: -8.5064764, 3.1202321, -7.2221880, 2.5724189, -11.0788956, 10.3424187
4: -4.4789262, 3.5923789, -3.5299795, 2.9892185, -7.4681444, 7.1223578

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0346686, upper bound: 3.0368455
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
time: 0.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.24 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.0773033, upper bound: 3.0834000
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.0513261, upper bound: 3.0530705
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.1297552, upper bound: 3.1359257
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.0946588, upper bound: 3.0946588
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.0982168, upper bound: 3.0943154
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.0982168, upper bound: 3.0943154
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.1069409, upper bound: 3.1018577
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.1069409, upper bound: 3.1018577
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.0482081, upper bound: 3.0514635
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.0482081, upper bound: 3.0514635
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.0346686, upper bound: 3.0359654
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.0346686, upper bound: 3.0368455
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.2097423, 1.4932493, -1.5857339, 1.7527800, -2.9625223, 3.0789831
1: -11.2694292, 2.9661100, -12.4515667, 3.7825937, -15.0520229, 15.4176769
2: -4.9715719, 3.3173807, -6.1709681, 3.9850879, -8.9566593, 9.4883490
3: -7.1621819, 2.4731677, -8.2178135, 2.9477344, -10.1099138, 10.6909809
4: -3.2760181, 2.8666625, -4.1961927, 3.3361864, -6.6122041, 7.0628548

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0513261, upper bound: 3.0530705
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0513261, upper bound: 3.0530705
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.1499847, 1.4499841, -1.5380487, 1.7190757, -2.8690603, 2.9880328
1: -10.9987278, 2.8335676, -12.4937382, 3.6779373, -14.6766653, 15.3273058
2: -4.7651887, 3.2024732, -6.0484319, 3.9286642, -8.6938534, 9.2509012
3: -6.9406676, 2.3919363, -8.1838322, 2.8904343, -9.8311024, 10.5757685
4: -3.1229849, 2.7919307, -4.0983024, 3.2566757, -6.3796601, 6.8902321

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0513261, upper bound: 3.0530705
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0513261, upper bound: 3.0530705
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.3914477, 1.5896201, -1.6964632, 1.8400882, -3.2315357, 3.2860832
1: -11.4863129, 3.3692830, -12.9356985, 4.0277605, -15.5140734, 16.3049812
2: -5.5162153, 3.5947320, -6.5428562, 4.2016149, -9.7178297, 10.1375885
3: -7.4902196, 2.6684668, -8.5972033, 3.1043184, -10.5945377, 11.2656698
4: -3.6932654, 3.0478096, -4.4759502, 3.4979689, -7.1912341, 7.5237598

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0946588, upper bound: 3.0946588
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0946588, upper bound: 3.0946588
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.3315125, 1.5452149, -1.6546468, 1.8205773, -3.1520896, 3.1998613
1: -11.1878996, 3.2369714, -12.9995623, 3.9364529, -15.1243515, 16.2365341
2: -5.3090534, 3.4759879, -6.4439631, 4.1567316, -9.4657822, 9.9199486
3: -7.2625647, 2.5835330, -8.5846148, 3.0545309, -10.3170958, 11.1681480
4: -3.5412560, 2.9661832, -4.3902826, 3.4392674, -6.9805217, 7.3564649

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0946588, upper bound: 3.0946588
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0946588, upper bound: 3.0946588
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.2097423, 1.4932493, -1.5155612, 1.7437913, -2.9535336, 3.0088105
1: -11.2694292, 2.9661100, -12.4957151, 3.5932887, -14.8627176, 15.4618254
2: -4.9715719, 3.3173807, -5.8347397, 3.8786392, -8.8502102, 9.1521196
3: -7.1621819, 2.4731677, -8.0386419, 2.8724000, -10.0345821, 10.5118084
4: -3.2760181, 2.8666625, -3.9789469, 3.3175647, -6.5935817, 6.8456097

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0724094, upper bound: 3.0669334
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0343026, upper bound: 3.0347200
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.2097423, 1.4932493, -1.7273169, 1.8819833, -3.0917251, 3.2205663
1: -11.2694292, 2.9661100, -12.8924522, 4.0498548, -15.3192844, 15.8585615
2: -4.9715719, 3.3173807, -6.4719362, 4.2366500, -9.2082195, 9.7893171
3: -7.1621819, 2.4731677, -8.5064764, 3.1202321, -10.2824125, 10.9796438
4: -3.2760181, 2.8666625, -4.4789262, 3.5923789, -6.8683963, 7.3455887

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0724094, upper bound: 3.0669334
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0343026, upper bound: 3.0347200
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.3914477, 1.5896201, -1.5155612, 1.7437913, -3.1352391, 3.1051812
1: -11.4863129, 3.3692830, -12.4957151, 3.5932887, -15.0796013, 15.8649979
2: -5.5162153, 3.5947320, -5.8347397, 3.8786392, -9.3948545, 9.4294691
3: -7.4902196, 2.6684668, -8.0386419, 2.8724000, -10.3626194, 10.7071085
4: -3.6932654, 3.0478096, -3.9789469, 3.3175647, -7.0108294, 7.0267563

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0718579, upper bound: 3.0669086
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0507674, upper bound: 3.0480210
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.3914477, 1.5896201, -1.7273169, 1.8819833, -3.2734306, 3.3169370
1: -11.4863129, 3.3692830, -12.8924522, 4.0498548, -15.5361671, 16.2617359
2: -5.5162153, 3.5947320, -6.4719362, 4.2366500, -9.7528639, 10.0666685
3: -7.4902196, 2.6684668, -8.5064764, 3.1202321, -10.6104517, 11.1749430
4: -3.6932654, 3.0478096, -4.4789262, 3.5923789, -7.2856441, 7.5267358

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0718579, upper bound: 3.0669086
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0507675, upper bound: 3.0486089
time: 0.44 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.5155612, 1.7437913, -1.1459812, 1.4534245, -2.9689856, 2.8897724
1: -12.4957151, 3.5932887, -11.0193834, 2.8110604, -15.3067760, 14.6126719
2: -5.8347397, 3.8786392, -4.7113099, 3.2089925, -9.0437307, 8.5899487
3: -8.0386419, 2.8724000, -6.8819108, 2.3776002, -10.4162407, 9.7543106
4: -3.9789469, 3.3175647, -3.1026425, 2.7976179, -6.7765646, 6.4202061

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
time: 0.46 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.4506932, 1.6929247, -1.1040034, 1.4078094, -2.8585026, 2.7969282
1: -12.2189274, 3.4503555, -10.7931662, 2.7167983, -14.9357243, 14.2435207
2: -5.6207881, 3.7536669, -4.5822306, 3.1243739, -8.7451620, 8.3358955
3: -7.8189039, 2.7826905, -6.7629304, 2.3225384, -10.1414423, 9.5456190
4: -3.8185954, 3.2230124, -3.0055897, 2.7083135, -6.5269089, 6.2286010

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.7273169, 1.8819833, -1.1459812, 1.4534245, -3.1807413, 3.0279639
1: -12.8924522, 4.0498548, -11.0193834, 2.8110604, -15.7035103, 15.0692387
2: -6.4719362, 4.2366500, -4.7113099, 3.2089925, -9.6809292, 8.9479589
3: -8.5064764, 3.1202321, -6.8819108, 2.3776002, -10.8840761, 10.0021420
4: -4.4789262, 3.5923789, -3.1026425, 2.7976179, -7.2765441, 6.6950207

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.6648276, 1.8310294, -1.1040034, 1.4078094, -3.0726371, 2.9350326
1: -12.6135674, 3.9137404, -10.7931662, 2.7167983, -15.3303633, 14.7069063
2: -6.2663879, 4.1135521, -4.5822306, 3.1243739, -9.3907604, 8.6957817
3: -8.2915897, 3.0326557, -6.7629304, 2.3225384, -10.6141281, 9.7955828
4: -4.3215246, 3.4963925, -3.0055897, 2.7083135, -7.0298381, 6.5019822

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.5155612, 1.7437913, -1.3085976, 1.5384215, -3.0539827, 3.0523889
1: -12.4957151, 3.5932887, -11.0998831, 3.1658449, -15.6615601, 14.6931715
2: -5.8347397, 3.8786392, -5.1915388, 3.4258237, -9.2605619, 9.0701780
3: -8.0386419, 2.8724000, -7.1353941, 2.5403614, -10.5790014, 10.0077944
4: -3.9789469, 3.3175647, -3.4749115, 2.9581385, -6.9370852, 6.7924757

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.4506932, 1.6929247, -1.2623985, 1.5108269, -2.9615202, 2.9553232
1: -12.2189274, 3.4503555, -11.1589785, 3.0705063, -15.2894335, 14.6093340
2: -5.6207881, 3.7536669, -5.0833993, 3.3887556, -9.0095434, 8.8370619
3: -7.8189039, 2.7826905, -7.1535316, 2.4998789, -10.3187828, 9.9362221
4: -3.8185954, 3.2230124, -3.3852398, 2.8973336, -6.7159286, 6.6082506

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
time: 0.42 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
time: 0.42 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.7273169, 1.8819833, -1.3085976, 1.5384215, -3.2657382, 3.1905801
1: -12.8924522, 4.0498548, -11.0998831, 3.1658449, -16.0582962, 15.1497383
2: -6.4719362, 4.2366500, -5.1915388, 3.4258237, -9.8977594, 9.4281874
3: -8.5064764, 3.1202321, -7.1353941, 2.5403614, -11.0468359, 10.2556257
4: -4.4789262, 3.5923789, -3.4749115, 2.9581385, -7.4370646, 7.0672903

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
time: 0.45 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.6648276, 1.8310294, -1.2623985, 1.5108269, -3.1756544, 3.0934272
1: -12.6135674, 3.9137404, -11.1589785, 3.0705063, -15.6840725, 15.0727186
2: -6.2663879, 4.1135521, -5.0833993, 3.3887556, -9.6551437, 9.1969481
3: -8.2915897, 3.0326557, -7.1535316, 2.4998789, -10.7914686, 10.1861868
4: -4.3215246, 3.4963925, -3.3852398, 2.8973336, -7.2188568, 6.8816323

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
time: 0.43 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
time: 0.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.20 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0513261, upper bound: 3.0530705
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0513261, upper bound: 3.0530705
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0513261, upper bound: 3.0530705
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0513261, upper bound: 3.0530705
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0946588, upper bound: 3.0946588
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0946588, upper bound: 3.0946588
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0946588, upper bound: 3.0946588
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0946588, upper bound: 3.0946588
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0724094, upper bound: 3.0669334
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0343026, upper bound: 3.0347200
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0724094, upper bound: 3.0669334
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0343026, upper bound: 3.0347200
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0718579, upper bound: 3.0669086
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0507674, upper bound: 3.0480210
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0718579, upper bound: 3.0669086
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0507675, upper bound: 3.0486089
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.1915734, 1.4786397, -1.5857339, 1.7527800, -2.9443533, 3.0643735
1: -11.1682863, 2.9246025, -12.4515667, 3.7825937, -14.9508791, 15.3761692
2: -4.9070315, 3.2798975, -6.1709681, 3.9850879, -8.8921194, 9.4508657
3: -7.0875225, 2.4465666, -8.2178135, 2.9477344, -10.0352564, 10.6643791
4: -3.2300665, 2.8400974, -4.1961927, 3.3361864, -6.5662527, 7.0362902

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0753726, upper bound: 3.0802420
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0753726, upper bound: 3.0834000
time: 0.47 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.1454144, 1.4319346, -1.5857339, 1.7527800, -2.8981943, 3.0176685
1: -11.0507336, 2.8164382, -12.4515667, 3.7825937, -14.8333273, 15.2680054
2: -4.7521067, 3.2045162, -6.1709681, 3.9850879, -8.7371941, 9.3754845
3: -6.9572916, 2.3847837, -8.2178135, 2.9477344, -9.9050255, 10.6025972
4: -3.1234121, 2.7428765, -4.1961927, 3.3361864, -6.4595981, 6.9390683

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0753726, upper bound: 3.0802420
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0753726, upper bound: 3.0834000
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.1910297, 1.4782270, -1.5380487, 1.7190757, -2.9101050, 3.0162759
1: -11.1674118, 2.9235811, -12.4937382, 3.6779373, -14.8453493, 15.4173183
2: -4.9061222, 3.2792010, -6.0484319, 3.9286642, -8.8347845, 9.3276329
3: -7.0866885, 2.4459028, -8.1838322, 2.8904343, -9.9771214, 10.6297331
4: -3.2291548, 2.8394578, -4.0983024, 3.2566757, -6.4858303, 6.9377599

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0439648, upper bound: 3.0439648
time: 0.46 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0439648, upper bound: 3.0530705
time: 0.43 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.1443491, 1.4307219, -1.5380487, 1.7190757, -2.8634245, 2.9687705
1: -11.0487347, 2.8145113, -12.4937382, 3.6779373, -14.7266722, 15.3082495
2: -4.7495809, 3.2026992, -6.0484319, 3.9286642, -8.6782446, 9.2511311
3: -6.9552937, 2.3831768, -8.1838322, 2.8904343, -9.8457279, 10.5670080
4: -3.1212478, 2.7410121, -4.0983024, 3.2566757, -6.3779235, 6.8393145

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0439648, upper bound: 3.0439648
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0439648, upper bound: 3.0530705
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.3667836, 1.5709029, -1.6964632, 1.8400882, -3.2068715, 3.2673657
1: -11.3657093, 3.3136542, -12.9356985, 4.0277605, -15.3934698, 16.2493534
2: -5.4319611, 3.5435553, -6.5428562, 4.2016149, -9.6335745, 10.0864105
3: -7.3961177, 2.6334047, -8.5972033, 3.1043184, -10.5004358, 11.2306080
4: -3.6333270, 3.0123208, -4.4759502, 3.4979689, -7.1312957, 7.4882708

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0800020, upper bound: 3.0847196
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0800020, upper bound: 3.1359257
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.3202276, 1.5406631, -1.6964632, 1.8400882, -3.1603160, 3.2371261
1: -11.4319592, 3.2100420, -12.9356985, 4.0277605, -15.4597197, 16.1457405
2: -5.2961969, 3.4978967, -6.5428562, 4.2016149, -9.4978113, 10.0407524
3: -7.3710346, 2.5808082, -8.5972033, 3.1043184, -10.4753532, 11.1780109
4: -3.5316677, 2.9448128, -4.4759502, 3.4979689, -7.0296359, 7.4207625

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0800020, upper bound: 3.0847196
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0800020, upper bound: 3.1359257
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.3661896, 1.5705174, -1.6546468, 1.8205773, -3.1867669, 3.2251642
1: -11.3633757, 3.3125360, -12.9995623, 3.9364529, -15.2998266, 16.3120956
2: -5.4304166, 3.5425310, -6.4439631, 4.1567316, -9.5871468, 9.9864931
3: -7.3944383, 2.6326032, -8.5846148, 3.0545309, -10.4489689, 11.2172184
4: -3.6320860, 3.0115824, -4.3902826, 3.4392674, -7.0713534, 7.4018650

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0530705, upper bound: 3.0513261
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0530705, upper bound: 3.0946589
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.3202276, 1.5406631, -1.6546468, 1.8205773, -3.1408048, 3.1953094
1: -11.4319592, 3.2100420, -12.9995623, 3.9364529, -15.3684072, 16.2096043
2: -5.2961969, 3.4978967, -6.4439631, 4.1567316, -9.4529266, 9.9418573
3: -7.3710346, 2.5808082, -8.5846148, 3.0545309, -10.4255657, 11.1654224
4: -3.5316677, 2.9448128, -4.3902826, 3.4392674, -6.9709344, 7.3350949

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0439648, upper bound: 3.0513261
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0530705, upper bound: 3.0946589
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.1915734, 1.4786397, -1.5155612, 1.7437913, -2.9353647, 2.9942007
1: -11.1682863, 2.9246025, -12.4957151, 3.5932887, -14.7615747, 15.4203176
2: -4.9070315, 3.2798975, -5.8347397, 3.8786392, -8.7856693, 9.1146364
3: -7.0875225, 2.4465666, -8.0386419, 2.8724000, -9.9599228, 10.4852076
4: -3.2300665, 2.8400974, -3.9789469, 3.3175647, -6.5476308, 6.8190441

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0395703, upper bound: 3.0394135
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0395703, upper bound: 3.0394135
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.1454144, 1.4319346, -1.4506932, 1.6929247, -2.8383391, 2.8826280
1: -11.0507336, 2.8164382, -12.2189274, 3.4503555, -14.5010891, 15.0353661
2: -4.7521067, 3.2045162, -5.6207881, 3.7536669, -8.5057716, 8.8253040
3: -6.9572916, 2.3847837, -7.8189039, 2.7826905, -9.7399826, 10.2036877
4: -3.1234121, 2.7428765, -3.8185954, 3.2230124, -6.3464231, 6.5614719

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0395703, upper bound: 3.0394135
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0395703, upper bound: 3.0394135
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.1915734, 1.4786397, -1.7273169, 1.8819833, -3.0735564, 3.2059565
1: -11.1682863, 2.9246025, -12.8924522, 4.0498548, -15.2181406, 15.8170538
2: -4.9070315, 3.2798975, -6.4719362, 4.2366500, -9.1436806, 9.7518339
3: -7.0875225, 2.4465666, -8.5064764, 3.1202321, -10.2077541, 10.9530430
4: -3.2300665, 2.8400974, -4.4789262, 3.5923789, -6.8224454, 7.3190236

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0342633, upper bound: 3.0347200
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0342633, upper bound: 3.0347200
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.1454144, 1.4319346, -1.6648276, 1.8310294, -2.9764433, 3.0967622
1: -11.0507336, 2.8164382, -12.6135674, 3.9137404, -14.9644737, 15.4300060
2: -4.7521067, 3.2045162, -6.2663879, 4.1135521, -8.8656578, 9.4709044
3: -6.9572916, 2.3847837, -8.2915897, 3.0326557, -9.9899464, 10.6763735
4: -3.1234121, 2.7428765, -4.3215246, 3.4963925, -6.6198044, 7.0644007

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0342633, upper bound: 3.0347200
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0342633, upper bound: 3.0347200
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.3667836, 1.5709029, -1.5155612, 1.7437913, -3.1105750, 3.0864642
1: -11.3657093, 3.3136542, -12.4957151, 3.5932887, -14.9589977, 15.8093691
2: -5.4319611, 3.5435553, -5.8347397, 3.8786392, -9.3105993, 9.3782930
3: -7.3961177, 2.6334047, -8.0386419, 2.8724000, -10.2685175, 10.6720467
4: -3.6333270, 3.0123208, -3.9789469, 3.3175647, -6.9508915, 6.9912677

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0507674, upper bound: 3.0480210
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0507674, upper bound: 3.0480210
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.3202276, 1.5406631, -1.4506932, 1.6929247, -3.0131524, 2.9913564
1: -11.4319592, 3.2100420, -12.2189274, 3.4503555, -14.8823128, 15.4289694
2: -5.2961969, 3.4978967, -5.6207881, 3.7536669, -9.0498638, 9.1186848
3: -7.3710346, 2.5808082, -7.8189039, 2.7826905, -10.1537247, 10.3997116
4: -3.5316677, 2.9448128, -3.8185954, 3.2230124, -6.7546782, 6.7634082

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0507674, upper bound: 3.0480210
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0507674, upper bound: 3.0480210
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.3667836, 1.5709029, -1.7273169, 1.8819833, -3.2487664, 3.2982197
1: -11.3657093, 3.3136542, -12.8924522, 4.0498548, -15.4155636, 16.2061043
2: -5.4319611, 3.5435553, -6.4719362, 4.2366500, -9.6686087, 10.0154905
3: -7.3961177, 2.6334047, -8.5064764, 3.1202321, -10.5163498, 11.1398811
4: -3.6333270, 3.0123208, -4.4789262, 3.5923789, -7.2257061, 7.4912472

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0519122, upper bound: 3.0486089
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0519122, upper bound: 3.0486089
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.3202276, 1.5406631, -1.6648276, 1.8310294, -3.1512570, 3.2054906
1: -11.4319592, 3.2100420, -12.6135674, 3.9137404, -15.3456993, 15.8236084
2: -5.2961969, 3.4978967, -6.2663879, 4.1135521, -9.4097481, 9.7642841
3: -7.3710346, 2.5808082, -8.2915897, 3.0326557, -10.4036903, 10.8723984
4: -3.5316677, 2.9448128, -4.3215246, 3.4963925, -7.0280600, 7.2663374

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0519122, upper bound: 3.0486089
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0519122, upper bound: 3.0486089
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.4962656, 1.7277657, -1.1459812, 1.4534245, -2.9496894, 2.8737469
1: -12.3940687, 3.5498657, -11.0193834, 2.8110604, -15.2051296, 14.5692492
2: -5.7683768, 3.8385758, -4.7113099, 3.2089925, -8.9773693, 8.5498857
3: -7.9630532, 2.8440640, -6.8819108, 2.3776002, -10.3406515, 9.7259750
4: -3.9308376, 3.2871678, -3.1026425, 2.7976179, -6.7284555, 6.3898106

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0485565, upper bound: 3.0523338
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0485565, upper bound: 3.0523338
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.3481991, 1.6085140, -1.1459812, 1.4534245, -2.8016229, 2.7544949
1: -11.8539753, 3.2201543, -11.0193834, 2.8110604, -14.6650352, 14.2395363
2: -5.2833767, 3.5789263, -4.7113099, 3.2089925, -8.4923687, 8.2902365
3: -7.5355568, 2.6435511, -6.8819108, 2.3776002, -9.9131556, 9.5254622
4: -3.5737505, 3.0576921, -3.1026425, 2.7976179, -6.3713684, 6.1603336

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0485565, upper bound: 3.0523338
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0485565, upper bound: 3.0523338
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.4962656, 1.7277657, -1.1040034, 1.4078094, -2.9040749, 2.8317690
1: -12.3940687, 3.5498657, -10.7931662, 2.7167983, -15.1108656, 14.3430319
2: -5.7683768, 3.8385758, -4.5822306, 3.1243739, -8.8927488, 8.4208069
3: -7.9630532, 2.8440640, -6.7629304, 2.3225384, -10.2855911, 9.6069927
4: -3.9308376, 3.2871678, -3.0055897, 2.7083135, -6.6391511, 6.2927575

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
time: 0.46 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.3481991, 1.6085140, -1.1040034, 1.4078094, -2.7560081, 2.7125170
1: -11.8539753, 3.2201543, -10.7931662, 2.7167983, -14.5707722, 14.0133200
2: -5.2833767, 3.5789263, -4.5822306, 3.1243739, -8.4077511, 8.1611567
3: -7.5355568, 2.6435511, -6.7629304, 2.3225384, -9.8580952, 9.4064789
4: -3.5737505, 3.0576921, -3.0055897, 2.7083135, -6.2820640, 6.0632811

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.7001939, 1.8583646, -1.1459812, 1.4534245, -3.1536183, 3.0043459
1: -12.7454500, 3.9885948, -11.0193834, 2.8110604, -15.5565090, 15.0079765
2: -6.3774061, 4.1791201, -4.7113099, 3.2089925, -9.5863991, 8.8904305
3: -8.3987617, 3.0791073, -6.8819108, 2.3776002, -10.7763596, 9.9610176
4: -4.4105210, 3.5483694, -3.1026425, 2.7976179, -7.2081389, 6.6510115

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0482081, upper bound: 3.0514635
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0482081, upper bound: 3.0514635
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.5379350, 1.7419161, -1.1459812, 1.4534245, -2.9913588, 2.8878970
1: -12.2760038, 3.6307266, -11.0193834, 2.8110604, -15.0870647, 14.6501093
2: -5.8584676, 3.8938198, -4.7113099, 3.2089925, -9.0674601, 8.6051292
3: -7.9675727, 2.8662035, -6.8819108, 2.3776002, -10.3451719, 9.7481146
4: -4.0163546, 3.3032200, -3.1026425, 2.7976179, -6.8139725, 6.4058614

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0482081, upper bound: 3.0514635
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0482081, upper bound: 3.0514635
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.6995370, 1.8578225, -1.1040034, 1.4078094, -3.1073463, 2.9618256
1: -12.7426519, 3.9873409, -10.7931662, 2.7167983, -15.4594488, 14.7805071
2: -6.3755922, 4.1779008, -4.5822306, 3.1243739, -9.4999657, 8.7601309
3: -8.3966932, 3.0781417, -6.7629304, 2.3225384, -10.7192307, 9.8410702
4: -4.4090633, 3.5473762, -3.0055897, 2.7083135, -7.1173768, 6.5529661

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.5379350, 1.7419161, -1.1040034, 1.4078094, -2.9457440, 2.8459194
1: -12.2760038, 3.6307266, -10.7931662, 2.7167983, -14.9928007, 14.4238930
2: -5.8584676, 3.8938198, -4.5822306, 3.1243739, -8.9828415, 8.4760504
3: -7.9675727, 2.8662035, -6.7629304, 2.3225384, -10.2901115, 9.6291313
4: -4.0163546, 3.3032200, -3.0055897, 2.7083135, -6.7246680, 6.3088093

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.4962656, 1.7277657, -1.3085976, 1.5384215, -3.0346868, 3.0363631
1: -12.3940687, 3.5498657, -11.0998831, 3.1658449, -15.5599136, 14.6497488
2: -5.7683768, 3.8385758, -5.1915388, 3.4258237, -9.1942005, 9.0301151
3: -7.9630532, 2.8440640, -7.1353941, 2.5403614, -10.5034113, 9.9794579
4: -3.9308376, 3.2871678, -3.4749115, 2.9581385, -6.8889761, 6.7620792

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0346686, upper bound: 3.0359654
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0346687, upper bound: 3.0359654
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.3481991, 1.6085140, -1.3085976, 1.5384215, -2.8866200, 2.9171109
1: -11.8539753, 3.2201543, -11.0998831, 3.1658449, -15.0198202, 14.3200369
2: -5.2833767, 3.5789263, -5.1915388, 3.4258237, -8.7091999, 8.7704649
3: -7.5355568, 2.6435511, -7.1353941, 2.5403614, -10.0759163, 9.7789450
4: -3.5737505, 3.0576921, -3.4749115, 2.9581385, -6.5318890, 6.5326033

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0346686, upper bound: 3.0359654
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0346687, upper bound: 3.0359654
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.4962656, 1.7277657, -1.2623985, 1.5108269, -3.0070925, 2.9901643
1: -12.3940687, 3.5498657, -11.1589785, 3.0705063, -15.4645748, 14.7088442
2: -5.7683768, 3.8385758, -5.0833993, 3.3887556, -9.1571321, 8.9219751
3: -7.9630532, 2.8440640, -7.1535316, 2.4998789, -10.4629326, 9.9975958
4: -3.9308376, 3.2871678, -3.3852398, 2.8973336, -6.8281708, 6.6724076

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.3481991, 1.6085140, -1.2623985, 1.5108269, -2.8590260, 2.8709118
1: -11.8539753, 3.2201543, -11.1589785, 3.0705063, -14.9244814, 14.3791332
2: -5.2833767, 3.5789263, -5.0833993, 3.3887556, -8.6721325, 8.6623240
3: -7.5355568, 2.6435511, -7.1535316, 2.4998789, -10.0354357, 9.7970829
4: -3.5737505, 3.0576921, -3.3852398, 2.8973336, -6.4710841, 6.4429312

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.7001939, 1.8583646, -1.3085976, 1.5384215, -3.2386153, 3.1669621
1: -12.7454500, 3.9885948, -11.0998831, 3.1658449, -15.9112949, 15.0884771
2: -6.3774061, 4.1791201, -5.1915388, 3.4258237, -9.8032303, 9.3706589
3: -8.3987617, 3.0791073, -7.1353941, 2.5403614, -10.9391203, 10.2145004
4: -4.4105210, 3.5483694, -3.4749115, 2.9581385, -7.3686595, 7.0232811

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0346686, upper bound: 3.0368455
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0355464, upper bound: 3.0368455
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.5379350, 1.7419161, -1.3085976, 1.5384215, -3.0763555, 3.0505135
1: -12.2760038, 3.6307266, -11.0998831, 3.1658449, -15.4418488, 14.7306089
2: -5.8584676, 3.8938198, -5.1915388, 3.4258237, -9.2842913, 9.0853586
3: -7.9675727, 2.8662035, -7.1353941, 2.5403614, -10.5079308, 10.0015974
4: -4.0163546, 3.3032200, -3.4749115, 2.9581385, -6.9744930, 6.7781315

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0346686, upper bound: 3.0368455
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0355464, upper bound: 3.0368455
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.6995370, 1.8578225, -1.2623985, 1.5108269, -3.2103639, 3.1202209
1: -12.7426519, 3.9873409, -11.1589785, 3.0705063, -15.8131580, 15.1463194
2: -6.3755922, 4.1779008, -5.0833993, 3.3887556, -9.7643480, 9.2612982
3: -8.3966932, 3.0781417, -7.1535316, 2.4998789, -10.8965712, 10.2316732
4: -4.4090633, 3.5473762, -3.3852398, 2.8973336, -7.3063970, 6.9326162

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.5379350, 1.7419161, -1.2623985, 1.5108269, -3.0487614, 3.0043144
1: -12.2760038, 3.6307266, -11.1589785, 3.0705063, -15.3465099, 14.7897053
2: -5.8584676, 3.8938198, -5.0833993, 3.3887556, -9.2472229, 8.9772177
3: -7.9675727, 2.8662035, -7.1535316, 2.4998789, -10.4674501, 10.0197353
4: -4.0163546, 3.3032200, -3.3852398, 2.8973336, -6.9136868, 6.6884589

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
time: 0.60 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.48 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0753726, upper bound: 3.0802420
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0753726, upper bound: 3.0834000
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0753726, upper bound: 3.0802420
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0753726, upper bound: 3.0834000
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0439648, upper bound: 3.0439648
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0439648, upper bound: 3.0530705
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0439648, upper bound: 3.0439648
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0439648, upper bound: 3.0530705
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0800020, upper bound: 3.0847196
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0800020, upper bound: 3.1359257
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0800020, upper bound: 3.0847196
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0800020, upper bound: 3.1359257
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0530705, upper bound: 3.0513261
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0530705, upper bound: 3.0946589
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0439648, upper bound: 3.0513261
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0530705, upper bound: 3.0946589
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0395703, upper bound: 3.0394135
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0395703, upper bound: 3.0394135
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0395703, upper bound: 3.0394135
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0395703, upper bound: 3.0394135
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0342633, upper bound: 3.0347200
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0342633, upper bound: 3.0347200
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0342633, upper bound: 3.0347200
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0342633, upper bound: 3.0347200
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0507674, upper bound: 3.0480210
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0507674, upper bound: 3.0480210
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0507674, upper bound: 3.0480210
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0507674, upper bound: 3.0480210
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0519122, upper bound: 3.0486089
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0519122, upper bound: 3.0486089
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0519122, upper bound: 3.0486089
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0519122, upper bound: 3.0486089
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0485565, upper bound: 3.0523338
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0485565, upper bound: 3.0523338
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0485565, upper bound: 3.0523338
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0485565, upper bound: 3.0523338
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0350191, upper bound: 3.0350191
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0482081, upper bound: 3.0514635
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0482081, upper bound: 3.0514635
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0482081, upper bound: 3.0514635
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0482081, upper bound: 3.0514635
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0313148, upper bound: 3.0306745
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0346686, upper bound: 3.0359654
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0346687, upper bound: 3.0359654
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0346686, upper bound: 3.0359654
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0346687, upper bound: 3.0359654
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0306745, upper bound: 3.0313148
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0346686, upper bound: 3.0368455
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0355464, upper bound: 3.0368455
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0346686, upper bound: 3.0368455
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0355464, upper bound: 3.0368455
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -3.0275240, upper bound: 3.0275240

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.1915734, 1.4786397, -1.1915734, 1.4786397, -2.6702130, 2.6702130
1: -11.1682863, 2.9246025, -11.1682863, 2.9246025, -14.0928888, 14.0928888
2: -4.9070315, 3.2798975, -4.9070315, 3.2798975, -8.1869287, 8.1869287
3: -7.0875225, 2.4465666, -7.0875225, 2.4465666, -9.5340891, 9.5340891
4: -3.2300665, 2.8400974, -3.2300665, 2.8400974, -6.0701637, 6.0701637

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1188764, upper bound: 3.1216693
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1845377, upper bound: 3.1845377
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.1915734, 1.4786397, -1.3667836, 1.5709029, -2.7624760, 2.8454227
1: -11.1682863, 2.9246025, -11.3657093, 3.3136542, -14.4819403, 14.2903118
2: -4.9070315, 3.2798975, -5.4319611, 3.5435553, -8.4505863, 8.7118578
3: -7.0875225, 2.4465666, -7.3961177, 2.6334047, -9.7209272, 9.8426838
4: -3.2300665, 2.8400974, -3.6333270, 3.0123208, -6.2423873, 6.4734244

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1188764, upper bound: 3.1383506
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1845377, upper bound: 3.1919221
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.1454144, 1.4319346, -1.1915734, 1.4786397, -2.6240540, 2.6235080
1: -11.0507336, 2.8164382, -11.1682863, 2.9246025, -13.9753361, 13.9847240
2: -4.7521067, 3.2045162, -4.9070315, 3.2798975, -8.0320044, 8.1115475
3: -6.9572916, 2.3847837, -7.0875225, 2.4465666, -9.4038582, 9.4723063
4: -3.1234121, 2.7428765, -3.2300665, 2.8400974, -5.9635091, 5.9729428

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0338548, upper bound: 3.0424790
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0740149, upper bound: 3.0788234
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.1454144, 1.4319346, -1.3667836, 1.5709029, -2.7163172, 2.7987182
1: -11.0507336, 2.8164382, -11.3657093, 3.3136542, -14.3643875, 14.1821470
2: -4.7521067, 3.2045162, -5.4319611, 3.5435553, -8.2956619, 8.6364765
3: -6.9572916, 2.3847837, -7.3961177, 2.6334047, -9.5906963, 9.7809010
4: -3.1234121, 2.7428765, -3.6333270, 3.0123208, -6.1357327, 6.3762031

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0338548, upper bound: 3.0424790
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0740149, upper bound: 3.0821817
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.1910297, 1.4782270, -1.1167562, 1.4029787, -2.5940077, 2.5949831
1: -11.1674118, 2.9235811, -10.7593880, 2.7411087, -13.9085207, 13.6829681
2: -4.9061222, 3.2792010, -4.6143460, 3.1281855, -8.0343056, 7.8935471
3: -7.0866885, 2.4459028, -6.7617717, 2.3256044, -9.4122925, 9.2076740
4: -3.2291548, 2.8394578, -3.0389829, 2.6892343, -5.9183888, 5.8784404

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0220411, upper bound: 3.0154758
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0788234, upper bound: 3.0740149
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.1910297, 1.4782270, -1.3205316, 1.5410773, -2.7321067, 2.7987585
1: -11.1674118, 2.9235811, -11.4326458, 3.2106230, -14.3780346, 14.3562269
2: -4.9061222, 3.2792010, -5.2970490, 3.4985454, -8.4046669, 8.5762491
3: -7.0866885, 2.4459028, -7.3716960, 2.5813427, -9.6680317, 9.8175974
4: -3.2291548, 2.8394578, -3.5323834, 2.9455428, -6.1746979, 6.3718410

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0220411, upper bound: 3.0198233
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0788234, upper bound: 3.0779508
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.1443491, 1.4307219, -1.1167562, 1.4029787, -2.5473275, 2.5474775
1: -11.0487347, 2.8145113, -10.7593880, 2.7411087, -13.7898436, 13.5738993
2: -4.7495809, 3.2026992, -4.6143460, 3.1281855, -7.8777637, 7.8170452
3: -6.9552937, 2.3831768, -6.7617717, 2.3256044, -9.2808981, 9.1449480
4: -3.1212478, 2.7410121, -3.0389829, 2.6892343, -5.8104806, 5.7799950

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9919464, upper bound: 2.9967288
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0416352, upper bound: 3.0416352
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.1443491, 1.4307219, -1.3205316, 1.5410773, -2.6854262, 2.7512534
1: -11.0487347, 2.8145113, -11.4326458, 3.2106230, -14.2593575, 14.2471571
2: -4.7495809, 3.2026992, -5.2970490, 3.4985454, -8.2481251, 8.4997482
3: -6.9552937, 2.3831768, -7.3716960, 2.5813427, -9.5366364, 9.7548723
4: -3.1212478, 2.7410121, -3.5323834, 2.9455428, -6.0667906, 6.2733955

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9919464, upper bound: 3.0119088
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0416352, upper bound: 3.0498716
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.3667836, 1.5709029, -1.1915734, 1.4786397, -2.8454230, 2.7624760
1: -11.3657093, 3.3136542, -11.1682863, 2.9246025, -14.2903118, 14.4819403
2: -5.4319611, 3.5435553, -4.9070315, 3.2798975, -8.7118578, 8.4505863
3: -7.3961177, 2.6334047, -7.0875225, 2.4465666, -9.8426838, 9.7209263
4: -3.6333270, 3.0123208, -3.2300665, 2.8400974, -6.4734244, 6.2423873

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1806907, upper bound: 3.1682564
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1919220, upper bound: 3.1869483
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.3667836, 1.5709029, -1.3667836, 1.5709029, -2.9376860, 2.9376860
1: -11.3657093, 3.3136542, -11.3657093, 3.3136542, -14.6793633, 14.6793633
2: -5.4319611, 3.5435553, -5.4319611, 3.5435553, -8.9755144, 8.9755144
3: -7.3961177, 2.6334047, -7.3961177, 2.6334047, -10.0295219, 10.0295219
4: -3.6333270, 3.0123208, -3.6333270, 3.0123208, -6.6456480, 6.6456480

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1806907, upper bound: 3.1958328
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1919220, upper bound: 3.1965328
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.3202276, 1.5406631, -1.1915734, 1.4786397, -2.7988672, 2.7322364
1: -11.4319592, 3.2100420, -11.1682863, 2.9246025, -14.3565617, 14.3783274
2: -5.2961969, 3.4978967, -4.9070315, 3.2798975, -8.5760946, 8.4049282
3: -7.3710346, 2.5808082, -7.0875225, 2.4465666, -9.8176012, 9.6683311
4: -3.5316677, 2.9448128, -3.2300665, 2.8400974, -6.3717647, 6.1748791

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0338548, upper bound: 3.0533063
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0779507, upper bound: 3.0826488
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.3202276, 1.5406631, -1.3667836, 1.5709029, -2.8911304, 2.9074466
1: -11.4319592, 3.2100420, -11.3657093, 3.3136542, -14.7456131, 14.5757494
2: -5.2961969, 3.4978967, -5.4319611, 3.5435553, -8.8397522, 8.9298573
3: -7.3710346, 2.5808082, -7.3961177, 2.6334047, -10.0044394, 9.9769258
4: -3.5316677, 2.9448128, -3.6333270, 3.0123208, -6.5439882, 6.5781398

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0507451, upper bound: 3.1291448
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0779508, upper bound: 3.1169120
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.3661896, 1.5705174, -1.1167562, 1.4029787, -2.7691681, 2.6872735
1: -11.3633757, 3.3125360, -10.7593880, 2.7411087, -14.1044846, 14.0719242
2: -5.4304166, 3.5425310, -4.6143460, 3.1281855, -8.5586023, 8.1568775
3: -7.3944383, 2.6326032, -6.7617717, 2.3256044, -9.7200432, 9.3943739
4: -3.6320860, 3.0115824, -3.0389829, 2.6892343, -6.3213196, 6.0505652

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0534153, upper bound: 3.0494831
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0821817, upper bound: 3.0760900
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.3661896, 1.5705174, -1.3205316, 1.5410773, -2.9072669, 2.8910489
1: -11.3633757, 3.3125360, -11.4326458, 3.2106230, -14.5739985, 14.7451820
2: -5.4304166, 3.5425310, -5.2970490, 3.4985454, -8.9289618, 8.8395786
3: -7.3944383, 2.6326032, -7.3716960, 2.5813427, -9.9757805, 10.0042992
4: -3.6320860, 3.0115824, -3.5323834, 2.9455428, -6.5776291, 6.5439658

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0534153, upper bound: 3.1081283
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0821817, upper bound: 3.1110050
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.3202276, 1.5406631, -1.1167562, 1.4029787, -2.7232063, 2.6574190
1: -11.4319592, 3.2100420, -10.7593880, 2.7411087, -14.1730671, 13.9694300
2: -5.2961969, 3.4978967, -4.6143460, 3.1281855, -8.4243822, 8.1122427
3: -7.3710346, 2.5808082, -6.7617717, 2.3256044, -9.6966391, 9.3425798
4: -3.5316677, 2.9448128, -3.0389829, 2.6892343, -6.2209005, 5.9837956

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0351957, upper bound: 3.0299643
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0498716, upper bound: 3.0481518
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.3202276, 1.5406631, -1.3205316, 1.5410773, -2.8613048, 2.8611946
1: -11.4319592, 3.2100420, -11.4326458, 3.2106230, -14.6425819, 14.6426878
2: -5.2961969, 3.4978967, -5.2970490, 3.4985454, -8.7947426, 8.7949438
3: -7.3710346, 2.5808082, -7.3716960, 2.5813427, -9.9523773, 9.9525042
4: -3.5316677, 2.9448128, -3.5323834, 2.9455428, -6.4772100, 6.4771962

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0351957, upper bound: 3.0765787
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0498716, upper bound: 3.0719180
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.1915734, 1.4786397, -1.4962656, 1.7277657, -2.9193392, 2.9749048
1: -11.1682863, 2.9246025, -12.3940687, 3.5498657, -14.7181520, 15.3186712
2: -4.9070315, 3.2798975, -5.7683768, 3.8385758, -8.7456074, 9.0482740
3: -7.0875225, 2.4465666, -7.9630532, 2.8440640, -9.9315863, 10.4096184
4: -3.2300665, 2.8400974, -3.9308376, 3.2871678, -6.5172343, 6.7709351

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0232369, upper bound: 3.0162048
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0741186, upper bound: 3.0682086
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.1915734, 1.4786397, -1.3481991, 1.6085140, -2.8000870, 2.8268385
1: -11.1682863, 2.9246025, -11.8539753, 3.2201543, -14.3884401, 14.7785778
2: -4.9070315, 3.2798975, -5.2833767, 3.5789263, -8.4859581, 8.5632744
3: -7.0875225, 2.4465666, -7.5355568, 2.6435511, -9.7310724, 9.9821234
4: -3.2300665, 2.8400974, -3.5737505, 3.0576921, -6.2877584, 6.4138479

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0232369, upper bound: 3.0162048
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0741186, upper bound: 3.0682086
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.1454144, 1.4319346, -1.4962656, 1.7277657, -2.8731799, 2.9282002
1: -11.0507336, 2.8164382, -12.3940687, 3.5498657, -14.6005993, 15.2105064
2: -4.7521067, 3.2045162, -5.7683768, 3.8385758, -8.5906830, 8.9728928
3: -6.9572916, 2.3847837, -7.9630532, 2.8440640, -9.8013554, 10.3478365
4: -3.1234121, 2.7428765, -3.9308376, 3.2871678, -6.4105797, 6.6737142

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9841918, upper bound: 2.9872035
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0377213, upper bound: 3.0373765
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.1454144, 1.4319346, -1.3481991, 1.6085140, -2.7539277, 2.7801337
1: -11.0507336, 2.8164382, -11.8539753, 3.2201543, -14.2708874, 14.6704140
2: -4.7521067, 3.2045162, -5.2833767, 3.5789263, -8.3310328, 8.4878931
3: -6.9572916, 2.3847837, -7.5355568, 2.6435511, -9.6008425, 9.9203405
4: -3.1234121, 2.7428765, -3.5737505, 3.0576921, -6.1811032, 6.3166270

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9841918, upper bound: 2.9872035
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0377213, upper bound: 3.0373765
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.1915734, 1.4786397, -1.7001939, 1.8583646, -3.0499380, 3.1788335
1: -11.1682863, 2.9246025, -12.7454500, 3.9885948, -15.1568794, 15.6700525
2: -4.9070315, 3.2798975, -6.3774061, 4.1791201, -9.0861511, 9.6573038
3: -7.0875225, 2.4465666, -8.3987617, 3.0791073, -10.1666298, 10.8453264
4: -3.2300665, 2.8400974, -4.4105210, 3.5483694, -6.7784357, 7.2506185

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0196026, upper bound: 3.0102732
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0703345, upper bound: 3.0646726
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.1915734, 1.4786397, -1.5379350, 1.7419161, -2.9334893, 3.0165739
1: -11.1682863, 2.9246025, -12.2760038, 3.6307266, -14.7990112, 15.2006063
2: -4.9070315, 3.2798975, -5.8584676, 3.8938198, -8.8008518, 9.1383648
3: -7.0875225, 2.4465666, -7.9675727, 2.8662035, -9.9537258, 10.4141378
4: -3.2300665, 2.8400974, -4.0163546, 3.3032200, -6.5332861, 6.8564520

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0196026, upper bound: 3.0102732
time: 0.47 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0703345, upper bound: 3.0646726
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.1454144, 1.4319346, -1.6995370, 1.8578225, -3.0032368, 3.1314716
1: -11.0507336, 2.8164382, -12.7426519, 3.9873409, -15.0380745, 15.5590897
2: -4.7521067, 3.2045162, -6.3755922, 4.1779008, -8.9300070, 9.5801086
3: -6.9572916, 2.3847837, -8.3966932, 3.0781417, -10.0354328, 10.7814760
4: -3.1234121, 2.7428765, -4.4090633, 3.5473762, -6.6707883, 7.1519399

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9778782, upper bound: 2.9819944
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0309246, upper bound: 3.0311899
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.1454144, 1.4319346, -1.5379350, 1.7419161, -2.8873301, 2.9698696
1: -11.0507336, 2.8164382, -12.2760038, 3.6307266, -14.6814594, 15.0924416
2: -4.7521067, 3.2045162, -5.8584676, 3.8938198, -8.6459265, 9.0629835
3: -6.9572916, 2.3847837, -7.9675727, 2.8662035, -9.8234949, 10.3523550
4: -3.1234121, 2.7428765, -4.0163546, 3.3032200, -6.4266315, 6.7592306

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9778782, upper bound: 2.9819944
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0309246, upper bound: 3.0311899
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.3667836, 1.5709029, -1.4962656, 1.7277657, -3.0945492, 3.0671682
1: -11.3657093, 3.3136542, -12.3940687, 3.5498657, -14.9155750, 15.7077208
2: -5.4319611, 3.5435553, -5.7683768, 3.8385758, -9.2705364, 9.3119297
3: -7.3961177, 2.6334047, -7.9630532, 2.8440640, -10.2401819, 10.5964575
4: -3.6333270, 3.0123208, -3.9308376, 3.2871678, -6.9204950, 6.9431581

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0505827, upper bound: 3.0465498
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0738615, upper bound: 3.0682875
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.3667836, 1.5709029, -1.3481991, 1.6085140, -2.9752970, 2.9191015
1: -11.3657093, 3.3136542, -11.8539753, 3.2201543, -14.5858612, 15.1676292
2: -5.4319611, 3.5435553, -5.2833767, 3.5789263, -9.0108871, 8.8269320
3: -7.3961177, 2.6334047, -7.5355568, 2.6435511, -10.0396690, 10.1689615
4: -3.6333270, 3.0123208, -3.5737505, 3.0576921, -6.6910186, 6.5860710

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0505827, upper bound: 3.0465498
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0738615, upper bound: 3.0682875
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.3202276, 1.5406631, -1.4962656, 1.7277657, -3.0479932, 3.0369287
1: -11.4319592, 3.2100420, -12.3940687, 3.5498657, -14.9818249, 15.6041107
2: -5.2961969, 3.4978967, -5.7683768, 3.8385758, -9.1347733, 9.2662716
3: -7.3710346, 2.5808082, -7.9630532, 2.8440640, -10.2150984, 10.5438614
4: -3.5316677, 2.9448128, -3.9308376, 3.2871678, -6.8188353, 6.8756504

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0356516, upper bound: 3.0331832
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0459666, upper bound: 3.0438931
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.3202276, 1.5406631, -1.3481991, 1.6085140, -2.9287415, 2.8888619
1: -11.4319592, 3.2100420, -11.8539753, 3.2201543, -14.6521130, 15.0640163
2: -5.2961969, 3.4978967, -5.2833767, 3.5789263, -8.8751230, 8.7812729
3: -7.3710346, 2.5808082, -7.5355568, 2.6435511, -10.0145855, 10.1163654
4: -3.5316677, 2.9448128, -3.5737505, 3.0576921, -6.5893588, 6.5185633

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9841918, upper bound: 3.0331832
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0459666, upper bound: 3.0438931
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.3667836, 1.5709029, -1.7001939, 1.8583646, -3.2251482, 3.2710967
1: -11.3657093, 3.3136542, -12.7454500, 3.9885948, -15.3543043, 16.0591030
2: -5.4319611, 3.5435553, -6.3774061, 4.1791201, -9.6110802, 9.9209614
3: -7.3961177, 2.6334047, -8.3987617, 3.0791073, -10.4752254, 11.0321655
4: -3.6333270, 3.0123208, -4.4105210, 3.5483694, -7.1816964, 7.4228420

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0556452, upper bound: 3.0517839
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0696643, upper bound: 3.0646443
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.3667836, 1.5709029, -1.5379350, 1.7419161, -3.1086993, 3.1088371
1: -11.3657093, 3.3136542, -12.2760038, 3.6307266, -14.9964361, 15.5896549
2: -5.4319611, 3.5435553, -5.8584676, 3.8938198, -9.3257809, 9.4020233
3: -7.3961177, 2.6334047, -7.9675727, 2.8662035, -10.2623215, 10.6009769
4: -3.6333270, 3.0123208, -4.0163546, 3.3032200, -6.9365468, 7.0286751

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0556452, upper bound: 3.0517839
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0696643, upper bound: 3.0646443
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.3202276, 1.5406631, -1.6995370, 1.8578225, -3.1780500, 3.2401998
1: -11.4319592, 3.2100420, -12.7426519, 3.9873409, -15.4192963, 15.9526939
2: -5.2961969, 3.4978967, -6.3755922, 4.1779008, -9.4740982, 9.8734894
3: -7.3710346, 2.5808082, -8.3966932, 3.0781417, -10.4491768, 10.9775009
4: -3.5316677, 2.9448128, -4.4090633, 3.5473762, -7.0790434, 7.3538761

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0439459, upper bound: 3.0398420
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0456954, upper bound: 3.0428360
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.3202276, 1.5406631, -1.5379350, 1.7419161, -3.0621438, 3.0785975
1: -11.4319592, 3.2100420, -12.2760038, 3.6307266, -15.0626850, 15.4860439
2: -5.2961969, 3.4978967, -5.8584676, 3.8938198, -9.1900167, 9.3563643
3: -7.3710346, 2.5808082, -7.9675727, 2.8662035, -10.2372379, 10.5483809
4: -3.5316677, 2.9448128, -4.0163546, 3.3032200, -6.8348866, 6.9611673

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0439459, upper bound: 3.0398420
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0456954, upper bound: 3.0428360
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.4962656, 1.7277657, -1.1913553, 1.4783846, -2.9746504, 2.9191210
1: -12.3940687, 3.5498657, -11.1678181, 2.9241743, -15.3182430, 14.7176838
2: -5.7683768, 3.8385758, -4.9065180, 3.2794578, -9.0478334, 8.7450943
3: -7.9630532, 2.8440640, -7.0870600, 2.4461918, -10.4092426, 9.9311237
4: -3.9308376, 3.2871678, -3.2296429, 2.8396437, -6.7704811, 6.5168104

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0403400, upper bound: 3.0447706
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0980162, upper bound: 3.0980163
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.4962656, 1.7277657, -1.4577374, 1.6942223, -3.1904871, 3.1855030
1: -12.3940687, 3.5498657, -12.2102890, 3.4636223, -15.8576908, 15.7601547
2: -5.7683768, 3.8385758, -5.6446853, 3.7684321, -9.5368090, 9.4832611
3: -7.9630532, 2.8440640, -7.8233190, 2.7845240, -10.7475758, 10.6673813
4: -3.9308376, 3.2871678, -3.8440928, 3.2258062, -7.1566434, 7.1312609

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0403400, upper bound: 3.0447706
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0980162, upper bound: 3.0980163
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.3481991, 1.6085140, -1.1913553, 1.4783846, -2.8265839, 2.7998688
1: -11.8539753, 3.2201543, -11.1678181, 2.9241743, -14.7781496, 14.3879719
2: -5.2833767, 3.5789263, -4.9065180, 3.2794578, -8.5628347, 8.4854441
3: -7.5355568, 2.6435511, -7.0870600, 2.4461918, -9.9817467, 9.7306108
4: -3.5737505, 3.0576921, -3.2296429, 2.8396437, -6.4133940, 6.2873349

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0111764, upper bound: 3.0205245
time: 0.48 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0468709, upper bound: 3.0510407
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.3481991, 1.6085140, -1.4577374, 1.6942223, -3.0424213, 3.0662513
1: -11.8539753, 3.2201543, -12.2102890, 3.4636223, -15.3175974, 15.4304399
2: -5.2833767, 3.5789263, -5.6446853, 3.7684321, -9.0518093, 9.2236118
3: -7.5355568, 2.6435511, -7.8233190, 2.7845240, -10.3200798, 10.4668684
4: -3.5737505, 3.0576921, -3.8440928, 3.2258062, -6.7995567, 6.9017849

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0111764, upper bound: 3.0205245
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0468709, upper bound: 3.0510408
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.4962656, 1.7277657, -1.1439414, 1.4302942, -2.9265594, 2.8717070
1: -12.3940687, 3.5498657, -11.0479164, 2.8137374, -15.2078056, 14.5977821
2: -5.7683768, 3.8385758, -4.7487350, 3.2018821, -8.9702568, 8.5873108
3: -7.9630532, 2.8440640, -6.9544759, 2.3824883, -10.3455400, 9.7985401
4: -3.9308376, 3.2871678, -3.1205547, 2.7401922, -6.6710300, 6.4077225

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9949820, upper bound: 2.9959954
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0501936, upper bound: 3.0468547
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.4962656, 1.7277657, -1.3591478, 1.6154935, -3.1117589, 3.0869136
1: -12.3940687, 3.5498657, -11.9062424, 3.2441120, -15.6381788, 15.4561071
2: -5.7683768, 3.8385758, -5.3400316, 3.6116674, -9.3800430, 9.1786079
3: -7.9630532, 2.8440640, -7.5855713, 2.6527381, -10.6157913, 10.4296350
4: -3.9308376, 3.2871678, -3.6162229, 3.0690885, -6.9999261, 6.9033909

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9949820, upper bound: 2.9959954
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0501936, upper bound: 3.0468547
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.3481991, 1.6085140, -1.1439414, 1.4302942, -2.7784927, 2.7524550
1: -11.8539753, 3.2201543, -11.0479164, 2.8137374, -14.6677132, 14.2680702
2: -5.2833767, 3.5789263, -4.7487350, 3.2018821, -8.4852581, 8.3276615
3: -7.5355568, 2.6435511, -6.9544759, 2.3824883, -9.9180441, 9.5980263
4: -3.5737505, 3.0576921, -3.1205547, 2.7401922, -6.3139429, 6.1782465

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9801806, upper bound: 2.9853469
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0334626, upper bound: 3.0334626
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.3481991, 1.6085140, -1.3591478, 1.6154935, -2.9636924, 2.9676614
1: -11.8539753, 3.2201543, -11.9062424, 3.2441120, -15.0980864, 15.1263943
2: -5.2833767, 3.5789263, -5.3400316, 3.6116674, -8.8950443, 8.9189577
3: -7.5355568, 2.6435511, -7.5855713, 2.6527381, -10.1882954, 10.2291222
4: -3.5737505, 3.0576921, -3.6162229, 3.0690885, -6.6428390, 6.6739140

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9801806, upper bound: 2.9853469
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0334626, upper bound: 3.0334626
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.7001939, 1.8583646, -1.1913553, 1.4783846, -3.1785784, 3.0497198
1: -12.7454500, 3.9885948, -11.1678181, 2.9241743, -15.6696224, 15.1564112
2: -6.3774061, 4.1791201, -4.9065180, 3.2794578, -9.6568642, 9.0856380
3: -8.3987617, 3.0791073, -7.0870600, 2.4461918, -10.8449497, 10.1661673
4: -4.4105210, 3.5483694, -3.2296429, 2.8396437, -7.2501645, 6.7780123

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0413930, upper bound: 3.0456869
time: 0.51 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0656152, upper bound: 3.0673205
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.7001939, 1.8583646, -1.4577374, 1.6942223, -3.3944161, 3.3161020
1: -12.7454500, 3.9885948, -12.2102890, 3.4636223, -16.2090721, 16.1988831
2: -6.3774061, 4.1791201, -5.6446853, 3.7684321, -10.1458378, 9.8238049
3: -8.3987617, 3.0791073, -7.8233190, 2.7845240, -11.1832848, 10.9024248
4: -4.4105210, 3.5483694, -3.8440928, 3.2258062, -7.6363273, 7.3924623

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0413930, upper bound: 3.0456869
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0656152, upper bound: 3.0673205
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.5379350, 1.7419161, -1.1913553, 1.4783846, -3.0163195, 2.9332709
1: -12.2760038, 3.6307266, -11.1678181, 2.9241743, -15.2001781, 14.7985449
2: -5.8584676, 3.8938198, -4.9065180, 3.2794578, -9.1379251, 8.8003378
3: -7.9675727, 2.8662035, -7.0870600, 2.4461918, -10.4137611, 9.9532633
4: -4.0163546, 3.3032200, -3.2296429, 2.8396437, -6.8559980, 6.5328627

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0238491, upper bound: 3.0299612
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0459472, upper bound: 3.0494225
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.5379350, 1.7419161, -1.4577374, 1.6942223, -3.2321568, 3.1996536
1: -12.2760038, 3.6307266, -12.2102890, 3.4636223, -15.7396259, 15.8410149
2: -5.8584676, 3.8938198, -5.6446853, 3.7684321, -9.6268997, 9.5385056
3: -7.9675727, 2.8662035, -7.8233190, 2.7845240, -10.7520952, 10.6895199
4: -4.0163546, 3.3032200, -3.8440928, 3.2258062, -7.2421608, 7.1473131

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0238491, upper bound: 3.0299612
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0459472, upper bound: 3.0494225
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.6995370, 1.8578225, -1.1439414, 1.4302942, -3.1298308, 3.0017636
1: -12.7426519, 3.9873409, -11.0479164, 2.8137374, -15.5563889, 15.0352573
2: -6.3755922, 4.1779008, -4.7487350, 3.2018821, -9.5774736, 8.9266357
3: -8.3966932, 3.0781417, -6.9544759, 2.3824883, -10.7791796, 10.0326176
4: -4.4090633, 3.5473762, -3.1205547, 2.7401922, -7.1492558, 6.6679306

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9917596, upper bound: 2.9904796
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0311296, upper bound: 3.0296237
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.6995370, 1.8578225, -1.3591478, 1.6154935, -3.3150303, 3.2169700
1: -12.7426519, 3.9873409, -11.9062424, 3.2441120, -15.9867640, 15.8935814
2: -6.3755922, 4.1779008, -5.3400316, 3.6116674, -9.9872599, 9.5179310
3: -8.3966932, 3.0781417, -7.5855713, 2.6527381, -11.0494308, 10.6637135
4: -4.4090633, 3.5473762, -3.6162229, 3.0690885, -7.4781518, 7.1635990

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9917596, upper bound: 2.9904796
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0311296, upper bound: 3.0296237
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.5379350, 1.7419161, -1.1439414, 1.4302942, -2.9682286, 2.8858571
1: -12.2760038, 3.6307266, -11.0479164, 2.8137374, -15.0897408, 14.6786423
2: -5.8584676, 3.8938198, -4.7487350, 3.2018821, -9.0603495, 8.6425552
3: -7.9675727, 2.8662035, -6.9544759, 2.3824883, -10.3500586, 9.8206787
4: -4.0163546, 3.3032200, -3.1205547, 2.7401922, -6.7565465, 6.4237747

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9832750, upper bound: 2.9879965
time: 0.51 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0278778, upper bound: 3.0273295
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.5379350, 1.7419161, -1.3591478, 1.6154935, -3.1534281, 3.1010635
1: -12.2760038, 3.6307266, -11.9062424, 3.2441120, -15.5201139, 15.5369692
2: -5.8584676, 3.8938198, -5.3400316, 3.6116674, -9.4701347, 9.2338514
3: -7.9675727, 2.8662035, -7.5855713, 2.6527381, -10.6203108, 10.4517746
4: -4.0163546, 3.3032200, -3.6162229, 3.0690885, -7.0854430, 6.9194422

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9832750, upper bound: 2.9879965
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0278778, upper bound: 3.0273295
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.4962656, 1.7277657, -1.3666067, 1.5707053, -3.0669708, 3.0943723
1: -12.3940687, 3.5498657, -11.3653240, 3.3132951, -15.7073631, 14.9151888
2: -5.7683768, 3.8385758, -5.4315586, 3.5432065, -9.3115797, 9.2701330
3: -7.9630532, 2.8440640, -7.3957367, 2.6330972, -10.5961485, 10.2398005
4: -3.9308376, 3.2871678, -3.6330032, 3.0119350, -6.9427724, 6.9201708

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0336220, upper bound: 3.0368953
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0673205, upper bound: 3.0656152
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.4962656, 1.7277657, -1.6565256, 1.8130418, -3.3093069, 3.3842912
1: -12.3940687, 3.5498657, -12.5522661, 3.9035225, -16.2975903, 16.1021309
2: -5.7683768, 3.8385758, -6.2612610, 4.0897059, -9.8580809, 10.0998363
3: -7.9630532, 2.8440640, -8.2563467, 3.0152867, -10.9783392, 11.1004105
4: -3.9308376, 3.2871678, -4.3137326, 3.4616680, -7.3925047, 7.6008997

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0336220, upper bound: 3.0368953
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0673205, upper bound: 3.0656152
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.3481991, 1.6085140, -1.3666067, 1.5707053, -2.9189041, 2.9751201
1: -11.8539753, 3.2201543, -11.3653240, 3.3132951, -15.1672707, 14.5854759
2: -5.2833767, 3.5789263, -5.4315586, 3.5432065, -8.8265810, 9.0104837
3: -7.5355568, 2.6435511, -7.3957367, 2.6330972, -10.1686516, 10.0392876
4: -3.5737505, 3.0576921, -3.6330032, 3.0119350, -6.5856857, 6.6906953

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0020287, upper bound: 3.0102131
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0296237, upper bound: 3.0311309
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.3481991, 1.6085140, -1.6565256, 1.8130418, -3.1612403, 3.2650394
1: -11.8539753, 3.2201543, -12.5522661, 3.9035225, -15.7574968, 15.7724152
2: -5.2833767, 3.5789263, -6.2612610, 4.0897059, -9.3730831, 9.8401871
3: -7.5355568, 2.6435511, -8.2563467, 3.0152867, -10.5508423, 10.8998976
4: -3.5737505, 3.0576921, -4.3137326, 3.4616680, -7.0354185, 7.3714237

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0020288, upper bound: 3.0102131
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0296237, upper bound: 3.0311309
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.4962656, 1.7277657, -1.3202276, 1.5406631, -3.0369287, 3.0479932
1: -12.3940687, 3.5498657, -11.4319592, 3.2100420, -15.6041107, 14.9818249
2: -5.7683768, 3.8385758, -5.2961969, 3.4978967, -9.2662735, 9.1347733
3: -7.9630532, 2.8440640, -7.3710346, 2.5808082, -10.5438614, 10.2150984
4: -3.9308376, 3.2871678, -3.5316677, 2.9448128, -6.8756504, 6.8188353

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0035361, upper bound: 3.0023102
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0494226, upper bound: 3.0459473
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.4962656, 1.7277657, -1.5358464, 1.7260950, -3.2223601, 3.2636120
1: -12.3940687, 3.5498657, -12.1337013, 3.6281481, -16.0222168, 15.6835670
2: -5.7683768, 3.8385758, -5.8453598, 3.8735933, -9.6419687, 9.6839352
3: -7.9630532, 2.8440640, -7.8916593, 2.8478739, -10.8109274, 10.7357235
4: -3.9308376, 3.2871678, -4.0116491, 3.2739217, -7.2047596, 7.2988167

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0035361, upper bound: 3.0023101
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0494226, upper bound: 3.0459473
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.3481991, 1.6085140, -1.3202276, 1.5406631, -2.8888621, 2.9287415
1: -11.8539753, 3.2201543, -11.4319592, 3.2100420, -15.0640163, 14.6521120
2: -5.2833767, 3.5789263, -5.2961969, 3.4978967, -8.7812729, 8.8751230
3: -7.5355568, 2.6435511, -7.3710346, 2.5808082, -10.1163654, 10.0145855
4: -3.5737505, 3.0576921, -3.5316677, 2.9448128, -6.5185633, 6.5893588

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811164, upper bound: 2.9855285
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0273294, upper bound: 3.0278778
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.3481991, 1.6085140, -1.5358464, 1.7260950, -3.0742939, 3.1443598
1: -11.8539753, 3.2201543, -12.1337013, 3.6281481, -15.4821234, 15.3538532
2: -5.2833767, 3.5789263, -5.8453598, 3.8735933, -9.1569700, 9.4242859
3: -7.5355568, 2.6435511, -7.8916593, 2.8478739, -10.3834305, 10.5352106
4: -3.5737505, 3.0576921, -4.0116491, 3.2739217, -6.8476725, 7.0693412

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811165, upper bound: 2.9855286
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0273294, upper bound: 3.0278778
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.7001939, 1.8583646, -1.3666067, 1.5707053, -3.2708993, 3.2249711
1: -12.7454500, 3.9885948, -11.3653240, 3.3132951, -16.0587444, 15.3539181
2: -6.3774061, 4.1791201, -5.4315586, 3.5432065, -9.9206123, 9.6106768
3: -8.3987617, 3.0791073, -7.3957367, 2.6330972, -11.0318565, 10.4748440
4: -4.4105210, 3.5483694, -3.6330032, 3.0119350, -7.4224558, 7.1813726

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0402061, upper bound: 3.0430489
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0541455, upper bound: 3.0541455
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.7001939, 1.8583646, -1.6565256, 1.8130418, -3.5132356, 3.5148902
1: -12.7454500, 3.9885948, -12.5522661, 3.9035225, -16.6489716, 16.5408611
2: -6.3774061, 4.1791201, -6.2612610, 4.0897059, -10.4671116, 10.4403811
3: -8.3987617, 3.0791073, -8.2563467, 3.0152867, -11.4140463, 11.3354530
4: -4.4105210, 3.5483694, -4.3137326, 3.4616680, -7.8721881, 7.8621006

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0402061, upper bound: 3.0430489
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0541455, upper bound: 3.0541455
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.5379350, 1.7419161, -1.3666067, 1.5707053, -3.1086400, 3.1085227
1: -12.2760038, 3.6307266, -11.3653240, 3.3132951, -15.5892992, 14.9960499
2: -5.8584676, 3.8938198, -5.4315586, 3.5432065, -9.4016724, 9.3253775
3: -7.9675727, 2.8662035, -7.3957367, 2.6330972, -10.6006670, 10.2619400
4: -4.0163546, 3.3032200, -3.6330032, 3.0119350, -7.0282884, 6.9362230

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0200822, upper bound: 3.0257294
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0296237, upper bound: 3.0318304
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.5379350, 1.7419161, -1.6565256, 1.8130418, -3.3509760, 3.3984418
1: -12.2760038, 3.6307266, -12.5522661, 3.9035225, -16.1795235, 16.1829929
2: -5.8584676, 3.8938198, -6.2612610, 4.0897059, -9.9481735, 10.1550808
3: -7.9675727, 2.8662035, -8.2563467, 3.0152867, -10.9828577, 11.1225491
4: -4.0163546, 3.3032200, -4.3137326, 3.4616680, -7.4780226, 7.6169529

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0200822, upper bound: 3.0257294
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0303612, upper bound: 3.0318304
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.6995370, 1.8578225, -1.3202276, 1.5406631, -3.2401998, 3.1780500
1: -12.7426519, 3.9873409, -11.4319592, 3.2100420, -15.9526939, 15.4192982
2: -6.3755922, 4.1779008, -5.2961969, 3.4978967, -9.8734894, 9.4740973
3: -8.3966932, 3.0781417, -7.3710346, 2.5808082, -10.9775009, 10.4491768
4: -4.4090633, 3.5473762, -3.5316677, 2.9448128, -7.3538761, 7.0790439

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0092927, upper bound: 3.0087541
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0311296, upper bound: 3.0296237
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.6995370, 1.8578225, -1.5358464, 1.7260950, -3.4256320, 3.3936687
1: -12.7426519, 3.9873409, -12.1337013, 3.6281481, -16.3708000, 16.1210384
2: -6.3755922, 4.1779008, -5.8453598, 3.8735933, -10.2491846, 10.0232601
3: -8.3966932, 3.0781417, -7.8916593, 2.8478739, -11.2445650, 10.9698009
4: -4.4090633, 3.5473762, -4.0116491, 3.2739217, -7.6829853, 7.5590253

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0092927, upper bound: 3.0087541
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0311296, upper bound: 3.0296237
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.5379350, 1.7419161, -1.3202276, 1.5406631, -3.0785975, 3.0621438
1: -12.2760038, 3.6307266, -11.4319592, 3.2100420, -15.4860458, 15.0626831
2: -5.8584676, 3.8938198, -5.2961969, 3.4978967, -9.3563643, 9.1900167
3: -7.9675727, 2.8662035, -7.3710346, 2.5808082, -10.5483809, 10.2372379
4: -4.0163546, 3.3032200, -3.5316677, 2.9448128, -6.9611673, 6.8348866

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9989356, upper bound: 3.0005360
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0232758, upper bound: 3.0232758
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.5379350, 1.7419161, -1.5358464, 1.7260950, -3.2640295, 3.2777624
1: -12.2760038, 3.6307266, -12.1337013, 3.6281481, -15.9041500, 15.7644272
2: -5.8584676, 3.8938198, -5.8453598, 3.8735933, -9.7320604, 9.7391796
3: -7.9675727, 2.8662035, -7.8916593, 2.8478739, -10.8154459, 10.7578630
4: -4.0163546, 3.3032200, -4.0116491, 3.2739217, -7.2902765, 7.3148689

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9989356, upper bound: 3.0005360
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0232758, upper bound: 3.0232758
time: 0.55 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.19 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.1188764, upper bound: 3.1216693
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.1845377, upper bound: 3.1845377
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.1188764, upper bound: 3.1383506
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.1845377, upper bound: 3.1919221
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0338548, upper bound: 3.0424790
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0740149, upper bound: 3.0788234
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0338548, upper bound: 3.0424790
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0740149, upper bound: 3.0821817
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0220411, upper bound: 3.0154758
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0788234, upper bound: 3.0740149
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0220411, upper bound: 3.0198233
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0788234, upper bound: 3.0779508
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9919464, upper bound: 2.9967288
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0416352, upper bound: 3.0416352
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9919464, upper bound: 3.0119088
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0416352, upper bound: 3.0498716
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.1806907, upper bound: 3.1682564
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.1919220, upper bound: 3.1869483
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.1806907, upper bound: 3.1958328
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.1919220, upper bound: 3.1965328
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0338548, upper bound: 3.0533063
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0779507, upper bound: 3.0826488
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0507451, upper bound: 3.1291448
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0779508, upper bound: 3.1169120
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0534153, upper bound: 3.0494831
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0821817, upper bound: 3.0760900
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0534153, upper bound: 3.1081283
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0821817, upper bound: 3.1110050
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0351957, upper bound: 3.0299643
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0498716, upper bound: 3.0481518
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0351957, upper bound: 3.0765787
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0498716, upper bound: 3.0719180
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0232369, upper bound: 3.0162048
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0741186, upper bound: 3.0682086
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0232369, upper bound: 3.0162048
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0741186, upper bound: 3.0682086
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9841918, upper bound: 2.9872035
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0377213, upper bound: 3.0373765
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9841918, upper bound: 2.9872035
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0377213, upper bound: 3.0373765
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0196026, upper bound: 3.0102732
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0703345, upper bound: 3.0646726
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0196026, upper bound: 3.0102732
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0703345, upper bound: 3.0646726
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9778782, upper bound: 2.9819944
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0309246, upper bound: 3.0311899
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9778782, upper bound: 2.9819944
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0309246, upper bound: 3.0311899
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0505827, upper bound: 3.0465498
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0738615, upper bound: 3.0682875
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0505827, upper bound: 3.0465498
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0738615, upper bound: 3.0682875
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0356516, upper bound: 3.0331832
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0459666, upper bound: 3.0438931
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9841918, upper bound: 3.0331832
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0459666, upper bound: 3.0438931
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0556452, upper bound: 3.0517839
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0696643, upper bound: 3.0646443
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0556452, upper bound: 3.0517839
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0696643, upper bound: 3.0646443
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0439459, upper bound: 3.0398420
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0456954, upper bound: 3.0428360
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0439459, upper bound: 3.0398420
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0456954, upper bound: 3.0428360
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0403400, upper bound: 3.0447706
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0980162, upper bound: 3.0980163
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0403400, upper bound: 3.0447706
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0980162, upper bound: 3.0980163
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0111764, upper bound: 3.0205245
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0468709, upper bound: 3.0510407
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0111764, upper bound: 3.0205245
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0468709, upper bound: 3.0510408
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9949820, upper bound: 2.9959954
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0501936, upper bound: 3.0468547
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9949820, upper bound: 2.9959954
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0501936, upper bound: 3.0468547
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9801806, upper bound: 2.9853469
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0334626, upper bound: 3.0334626
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9801806, upper bound: 2.9853469
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0334626, upper bound: 3.0334626
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0413930, upper bound: 3.0456869
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0656152, upper bound: 3.0673205
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0413930, upper bound: 3.0456869
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0656152, upper bound: 3.0673205
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0238491, upper bound: 3.0299612
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0459472, upper bound: 3.0494225
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0238491, upper bound: 3.0299612
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0459472, upper bound: 3.0494225
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9917596, upper bound: 2.9904796
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0311296, upper bound: 3.0296237
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9917596, upper bound: 2.9904796
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0311296, upper bound: 3.0296237
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9832750, upper bound: 2.9879965
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0278778, upper bound: 3.0273295
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9832750, upper bound: 2.9879965
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0278778, upper bound: 3.0273295
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0336220, upper bound: 3.0368953
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0673205, upper bound: 3.0656152
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0336220, upper bound: 3.0368953
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0673205, upper bound: 3.0656152
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0020287, upper bound: 3.0102131
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0296237, upper bound: 3.0311309
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0020288, upper bound: 3.0102131
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0296237, upper bound: 3.0311309
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0035361, upper bound: 3.0023102
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0494226, upper bound: 3.0459473
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0035361, upper bound: 3.0023101
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0494226, upper bound: 3.0459473
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9811164, upper bound: 2.9855285
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0273294, upper bound: 3.0278778
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9811165, upper bound: 2.9855286
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0273294, upper bound: 3.0278778
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0402061, upper bound: 3.0430489
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0541455, upper bound: 3.0541455
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0402061, upper bound: 3.0430489
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0541455, upper bound: 3.0541455
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0200822, upper bound: 3.0257294
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0296237, upper bound: 3.0318304
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0200822, upper bound: 3.0257294
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0303612, upper bound: 3.0318304
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0092927, upper bound: 3.0087541
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0311296, upper bound: 3.0296237
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0092927, upper bound: 3.0087541
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0311296, upper bound: 3.0296237
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9989356, upper bound: 3.0005360
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0232758, upper bound: 3.0232758
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -2.9989356, upper bound: 3.0005360
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -3.0232758, upper bound: 3.0232758

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.7970945, 1.1366076, -1.0099517, 1.3409599, -2.1380544, 2.1465592
1: -9.1047134, 2.0115535, -10.4073400, 2.5156114, -11.6203241, 12.4188938
2: -3.4911079, 2.4991722, -4.2757792, 2.9629648, -6.4540715, 6.7749510
3: -5.3678417, 1.8765234, -6.3897643, 2.1975868, -7.5654278, 8.2662849
4: -2.2343647, 2.2230315, -2.7650223, 2.6024613, -4.8368263, 4.9880528

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1187033, upper bound: 3.1214586
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1177383, upper bound: 3.1200518
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.1383243, 1.4370885, -1.1731349, 1.4643019, -2.6026263, 2.6102233
1: -10.9247408, 2.8038428, -11.0795002, 2.8828106, -13.8075514, 13.8833427
2: -4.7305412, 3.1777730, -4.8458300, 3.2449245, -7.9754658, 8.0236015
3: -6.8915405, 2.3700418, -7.0209394, 2.4202754, -9.3118162, 9.3909798
4: -3.1007495, 2.7679157, -3.1848581, 2.8151679, -5.9159174, 5.9527731

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1512041, upper bound: 3.1560687
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1401964, upper bound: 3.1401964
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.7970945, 1.1366076, -1.1726022, 1.4214875, -2.2185819, 2.3092096
1: -9.1047134, 2.0115535, -10.4738464, 2.8779371, -11.9826508, 12.4854002
2: -3.4911079, 2.4991722, -4.7513676, 3.1555865, -6.6466942, 7.2505388
3: -5.3678417, 1.8765234, -6.6485023, 2.3522940, -7.7201343, 8.5250235
4: -2.2343647, 2.2230315, -3.1365902, 2.7497952, -4.9841595, 5.3596220

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1297528, upper bound: 3.1382878
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1297527, upper bound: 3.1382811
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.1383243, 1.4370885, -1.3446057, 1.5543120, -2.6926363, 2.7816937
1: -10.9247408, 2.8038428, -11.2585287, 3.2624943, -14.1872339, 14.0623713
2: -4.7305412, 3.1777730, -5.3566713, 3.4997976, -8.2303371, 8.5344439
3: -6.8915405, 2.3700418, -7.3147697, 2.6007752, -9.4923143, 9.6848116
4: -3.1007495, 2.7679157, -3.5793447, 2.9816153, -6.0823646, 6.3472595

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1682564, upper bound: 3.1806907
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1682564, upper bound: 3.1919221
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.7735340, 1.0927823, -1.0099517, 1.3409599, -2.1144938, 2.1027339
1: -8.6353846, 1.9463543, -10.4073400, 2.5156114, -11.1509953, 12.3536940
2: -3.3554802, 2.4127922, -4.2757792, 2.9629648, -6.3184438, 6.6885715
3: -5.1615577, 1.8109016, -6.3897643, 2.1975868, -7.3591442, 8.2006655
4: -2.1475105, 2.1514876, -2.7650223, 2.6024613, -4.7499714, 4.9165096

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0336039, upper bound: 3.0431105
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0290481, upper bound: 3.0388937
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.0919143, 1.3894186, -1.1731349, 1.4643019, -2.5562162, 2.5625534
1: -10.7838097, 2.6956663, -11.0795002, 2.8828106, -13.6666203, 13.7751665
2: -4.5721989, 3.0989673, -4.8458300, 3.2449245, -7.8171229, 7.9447956
3: -6.7559719, 2.3078649, -7.0209394, 2.4202754, -9.1762476, 9.3288021
4: -2.9928617, 2.6681583, -3.1848581, 2.8151679, -5.8080292, 5.8530149

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0647195, upper bound: 3.0709187
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0519419, upper bound: 3.0548817
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.7735340, 1.0927823, -1.1726022, 1.4214875, -2.1950214, 2.2653840
1: -8.6353846, 1.9463543, -10.4738464, 2.8779371, -11.5133200, 12.4201994
2: -3.3554802, 2.4127922, -4.7513676, 3.1555865, -6.5110664, 7.1641598
3: -5.1615577, 1.8109016, -6.6485023, 2.3522940, -7.5138507, 8.4594040
4: -2.1475105, 2.1514876, -3.1365902, 2.7497952, -4.8973045, 5.2880778

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0330117, upper bound: 3.0419454
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0306685, upper bound: 3.0400301
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.0919143, 1.3894186, -1.3446057, 1.5543120, -2.6462264, 2.7340243
1: -10.7838097, 2.6956663, -11.2585287, 3.2624943, -14.0463037, 13.9541950
2: -4.5721989, 3.0989673, -5.3566713, 3.4997976, -8.0719948, 8.4556379
3: -6.7559719, 2.3078649, -7.3147697, 2.6007752, -9.3567467, 9.6226339
4: -2.9928617, 2.6681583, -3.5793447, 2.9816153, -5.9744759, 6.2475023

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0494831, upper bound: 3.0534153
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0494831, upper bound: 3.0821817
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.7970945, 1.1366076, -0.9500291, 1.2706285, -2.0677226, 2.0866365
1: -9.1047134, 2.0115535, -9.9069176, 2.3616753, -11.4663877, 11.9184685
2: -3.4911079, 2.4991722, -4.0343709, 2.7981148, -6.2892222, 6.5335426
3: -5.3678417, 1.8765234, -6.1281958, 2.0907111, -7.4585524, 8.0047178
4: -2.2343647, 2.2230315, -2.6114497, 2.4647799, -4.6991444, 4.8344812

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9980599, upper bound: 2.9965305
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9980599, upper bound: 3.0146186
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.1382445, 1.4370341, -1.0995879, 1.3893962, -2.5276406, 2.5366213
1: -10.9246197, 2.8036959, -10.6780634, 2.7023947, -13.6270142, 13.4817591
2: -4.7304206, 3.1776767, -4.5574398, 3.0951476, -7.8255682, 7.7351165
3: -6.8914251, 2.3699541, -6.7005434, 2.3013453, -9.1927700, 9.0704975
4: -3.1006255, 2.7678316, -2.9975276, 2.6655512, -5.7661767, 5.7653589

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0437378, upper bound: 3.0345316
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0437378, upper bound: 3.0740149
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.7970945, 1.1366076, -1.1349080, 1.4003193, -2.1974137, 2.2715154
1: -9.1047134, 2.0115535, -10.4769850, 2.7953627, -11.9000759, 12.4885387
2: -3.4911079, 2.4991722, -4.6512942, 3.1288595, -6.6199665, 7.1504650
3: -5.3678417, 1.8765234, -6.6609640, 2.3144982, -7.6823392, 8.5374861
4: -2.2343647, 2.2230315, -3.0605192, 2.6968501, -4.9312143, 5.2835503

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0221925, upper bound: 3.0198233
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0221925, upper bound: 3.0198233
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.1382445, 1.4370341, -1.2991329, 1.5251652, -2.6634097, 2.7361662
1: -10.9246197, 2.8036959, -11.3297234, 3.1618137, -14.0864334, 14.1334190
2: -4.7304206, 3.1776767, -5.2254505, 3.4568017, -8.1872215, 8.4031277
3: -6.8914251, 2.3699541, -7.2934699, 2.5502114, -9.4416351, 9.6634226
4: -3.1006255, 2.7678316, -3.4804327, 2.9168844, -6.0175099, 6.2482643

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0533063, upper bound: 3.0507451
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0533063, upper bound: 3.0779508
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.7735340, 1.0927823, -0.9500291, 1.2706285, -2.0441625, 2.0428114
1: -8.6353846, 1.9463543, -9.9069176, 2.3616753, -10.9970579, 11.8532696
2: -3.3554802, 2.4127922, -4.0343709, 2.7981148, -6.1535945, 6.4471631
3: -5.1615577, 1.8109016, -6.1281958, 2.0907111, -7.2522683, 7.9390974
4: -2.1475105, 2.1514876, -2.6114497, 2.4647799, -4.6122904, 4.7629375

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9897867, upper bound: 2.9945209
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9908305, upper bound: 2.9958101
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.0914965, 1.3889042, -1.0995879, 1.3893962, -2.4808924, 2.4884918
1: -10.7830114, 2.6950121, -10.6780634, 2.7023947, -13.4854050, 13.3730745
2: -4.5711455, 3.0982189, -4.5574398, 3.0951476, -7.6662931, 7.6556587
3: -6.7551632, 2.3073444, -6.7005434, 2.3013453, -9.0565090, 9.0078869
4: -2.9919815, 2.6673815, -2.9975276, 2.6655512, -5.6575327, 5.6649084

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9967288, upper bound: 2.9919464
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9967288, upper bound: 3.0416352
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.7735340, 1.0927823, -1.1349080, 1.4003193, -2.1738532, 2.2276900
1: -8.6353846, 1.9463543, -10.4769850, 2.7953627, -11.4307461, 12.4233379
2: -3.3554802, 2.4127922, -4.6512942, 3.1288595, -6.4843392, 7.0640864
3: -5.1615577, 1.8109016, -6.6609640, 2.3144982, -7.4760551, 8.4718657
4: -2.1475105, 2.1514876, -3.0605192, 2.6968501, -4.8443599, 5.2120066

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0031855, upper bound: 3.0119088
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0031855, upper bound: 3.0119088
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.0914965, 1.3889042, -1.2991329, 1.5251652, -2.6166615, 2.6880369
1: -10.7830114, 2.6950121, -11.3297234, 3.1618137, -13.9448242, 14.0247345
2: -4.5711455, 3.0982189, -5.2254505, 3.4568017, -8.0279474, 8.3236685
3: -6.7551632, 2.3073444, -7.2934699, 2.5502114, -9.3053741, 9.6008148
4: -2.9919815, 2.6673815, -3.4804327, 2.9168844, -5.9088659, 6.1478143

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9967288, upper bound: 3.0351957
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0299643, upper bound: 3.0498717
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.9106327, 1.1853050, -1.0099517, 1.3409599, -2.2515926, 2.1952567
1: -8.9279690, 2.2690125, -10.4073400, 2.5156114, -11.4435797, 12.6763525
2: -3.7429752, 2.6254594, -4.2757792, 2.9629648, -6.7059383, 6.9012384
3: -5.4113026, 1.9600556, -6.3897643, 2.1975868, -7.6088891, 8.3498192
4: -2.4330649, 2.3252549, -2.7650223, 2.6024613, -5.0355263, 5.0902758

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1805420, upper bound: 3.1682564
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1683225, upper bound: 3.1565417
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.3044778, 1.5242738, -1.1731349, 1.4643019, -2.7687798, 2.6974087
1: -11.0584764, 3.1699493, -11.0795002, 2.8828106, -13.9412851, 14.2494497
2: -5.2195768, 3.4209330, -4.8458300, 3.2449245, -8.4645014, 8.2667627
3: -7.1646547, 2.5415988, -7.0209394, 2.4202754, -9.5849304, 9.5625362
4: -3.4811053, 2.9271264, -3.1848581, 2.8151679, -6.2962732, 6.1119828

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1885167, upper bound: 3.1807516
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1696374, upper bound: 3.1567870
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.9106327, 1.1853050, -1.1726022, 1.4214875, -2.3321202, 2.3579068
1: -8.9279690, 2.2690125, -10.4738464, 2.8779371, -11.8059063, 12.7428589
2: -3.7429752, 2.6254594, -4.7513676, 3.1555865, -6.8985605, 7.3768272
3: -5.4113026, 1.9600556, -6.6485023, 2.3522940, -7.7635961, 8.6085577
4: -2.4330649, 2.3252549, -3.1365902, 2.7497952, -5.1828604, 5.4618449

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1187033, upper bound: 3.1950382
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1954439, upper bound: 3.1958328
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.3044778, 1.5242738, -1.3446057, 1.5543120, -2.8587899, 2.8688788
1: -11.0584764, 3.1699493, -11.2585287, 3.2624943, -14.3209677, 14.4284782
2: -5.2195768, 3.4209330, -5.3566713, 3.4997976, -8.7193737, 8.7776041
3: -7.1646547, 2.5415988, -7.3147697, 2.6007752, -9.7654295, 9.8563671
4: -3.4811053, 2.9271264, -3.5793447, 2.9816153, -6.4627204, 6.5064707

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1955478, upper bound: 3.1953546
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1955478, upper bound: 3.1965328
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.9023529, 1.1818460, -1.0099517, 1.3409599, -2.2433128, 2.1917977
1: -8.8204126, 2.2459600, -10.4073400, 2.5156114, -11.3360233, 12.6533003
2: -3.7143652, 2.6084802, -4.2757792, 2.9629648, -6.6773291, 6.8842592
3: -5.4461040, 1.9300833, -6.3897643, 2.1975868, -7.6436911, 8.3198452
4: -2.4172771, 2.3120294, -2.7650223, 2.6024613, -5.0197382, 5.0770502

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0504639, upper bound: 3.0529018
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0476112, upper bound: 3.0489499
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.2575183, 1.4950199, -1.1731349, 1.4643019, -2.7218204, 2.6681547
1: -11.1332922, 3.0673094, -11.0795002, 2.8828106, -14.0161028, 14.1468096
2: -5.0876622, 3.3761044, -4.8458300, 3.2449245, -8.3325863, 8.2219343
3: -7.1426382, 2.4905334, -7.0209394, 2.4202754, -9.5629139, 9.5114727
4: -3.3797951, 2.8625700, -3.1848581, 2.8151679, -6.1949630, 6.0474272

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0775704, upper bound: 3.0822010
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0519419, upper bound: 3.0605935
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.9023529, 1.1818460, -1.1726022, 1.4214875, -2.3238404, 2.3544476
1: -8.8204126, 2.2459600, -10.4738464, 2.8779371, -11.6983490, 12.7198067
2: -3.7143652, 2.6084802, -4.7513676, 3.1555865, -6.8699517, 7.3598475
3: -5.4461040, 1.9300833, -6.6485023, 2.3522940, -7.7983980, 8.5785837
4: -2.4172771, 2.3120294, -3.1365902, 2.7497952, -5.1670723, 5.4486198

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0810269, upper bound: 3.0861956
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1240333, upper bound: 3.1291353
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.2575183, 1.4950199, -1.3446057, 1.5543120, -2.8118303, 2.8396246
1: -11.1332922, 3.0673094, -11.2585287, 3.2624943, -14.3957863, 14.3258381
2: -5.0876622, 3.3761044, -5.3566713, 3.4997976, -8.5874586, 8.7327757
3: -7.1426382, 2.4905334, -7.3147697, 2.6007752, -9.7434130, 9.8053036
4: -3.3797951, 2.8625700, -3.5793447, 2.9816153, -6.3614101, 6.4419146

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1034910, upper bound: 3.1072570
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1034909, upper bound: 3.1169119
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.9106327, 1.1853050, -0.9500291, 1.2706285, -2.1812611, 2.1353340
1: -8.9279690, 2.2690125, -9.9069176, 2.3616753, -11.2896433, 12.1759272
2: -3.7429752, 2.6254594, -4.0343709, 2.7981148, -6.5410900, 6.6598301
3: -5.4113026, 1.9600556, -6.1281958, 2.0907111, -7.5020137, 8.0882502
4: -2.4330649, 2.3252549, -2.6114497, 2.4647799, -4.8978448, 4.9367037

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0530948, upper bound: 3.0489419
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0520994, upper bound: 3.0482157
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.3044778, 1.5242738, -1.0995879, 1.3893962, -2.6938736, 2.6238611
1: -11.0584764, 3.1699493, -10.6780634, 2.7023947, -13.7608690, 13.8480120
2: -5.2195768, 3.4209330, -4.5574398, 3.0951476, -8.3147240, 7.9783726
3: -7.1646547, 2.5415988, -6.7005434, 2.3013453, -9.4659996, 9.2421408
4: -3.4811053, 2.9271264, -2.9975276, 2.6655512, -6.1466565, 5.9246535

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0424790, upper bound: 3.0338548
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0424790, upper bound: 3.0760900
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.9106327, 1.1853050, -1.1349080, 1.4003193, -2.3109519, 2.3202128
1: -8.9279690, 2.2690125, -10.4769850, 2.7953627, -11.7233315, 12.7459974
2: -3.7429752, 2.6254594, -4.6512942, 3.1288595, -6.8718348, 7.2767534
3: -5.4113026, 1.9600556, -6.6609640, 2.3144982, -7.7258005, 8.6210194
4: -2.4330649, 2.3252549, -3.0605192, 2.6968501, -5.1299152, 5.3857737

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1129832, upper bound: 3.1081283
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1129832, upper bound: 3.1081283
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.3044778, 1.5242738, -1.2991329, 1.5251652, -2.8296430, 2.8234062
1: -11.0584764, 3.1699493, -11.3297234, 3.1618137, -14.2202883, 14.4996729
2: -5.2195768, 3.4209330, -5.2254505, 3.4568017, -8.6763783, 8.6463833
3: -7.1646547, 2.5415988, -7.2934699, 2.5502114, -9.7148666, 9.8350677
4: -3.4811053, 2.9271264, -3.4804327, 2.9168844, -6.3979897, 6.4075589

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0424790, upper bound: 3.1110050
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1164360, upper bound: 3.1110050
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.9023529, 1.1818460, -0.9500291, 1.2706285, -2.1729813, 2.1318750
1: -8.8204126, 2.2459600, -9.9069176, 2.3616753, -11.1820860, 12.1528759
2: -3.7143652, 2.6084802, -4.0343709, 2.7981148, -6.5124798, 6.6428509
3: -5.4461040, 1.9300833, -6.1281958, 2.0907111, -7.5368152, 8.0582771
4: -2.4172771, 2.3120294, -2.6114497, 2.4647799, -4.8820572, 4.9234781

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0192459, upper bound: 3.0180869
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9908305, upper bound: 3.0289540
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.2575183, 1.4950199, -1.0995879, 1.3893962, -2.6469140, 2.5946071
1: -11.1332922, 3.0673094, -10.6780634, 2.7023947, -13.8356867, 13.7453728
2: -5.0876622, 3.3761044, -4.5574398, 3.0951476, -8.1828098, 7.9335442
3: -7.1426382, 2.4905334, -6.7005434, 2.3013453, -9.4439831, 9.1910763
4: -3.3797951, 2.8625700, -2.9975276, 2.6655512, -6.0453463, 5.8600979

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0119088, upper bound: 3.0031855
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0119088, upper bound: 3.0481518
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.9023529, 1.1818460, -1.1349080, 1.4003193, -2.3026721, 2.3167539
1: -8.8204126, 2.2459600, -10.4769850, 2.7953627, -11.6157751, 12.7229452
2: -3.7143652, 2.6084802, -4.6512942, 3.1288595, -6.8432245, 7.2597737
3: -5.4461040, 1.9300833, -6.6609640, 2.3144982, -7.7606020, 8.5910463
4: -2.4172771, 2.3120294, -3.0605192, 2.6968501, -5.1141272, 5.3725476

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0719180, upper bound: 3.0719180
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0719180, upper bound: 3.0719180
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.2575183, 1.4950199, -1.2991329, 1.5251652, -2.7826834, 2.7941520
1: -11.1332922, 3.0673094, -11.3297234, 3.1618137, -14.2951059, 14.3970327
2: -5.0876622, 3.3761044, -5.2254505, 3.4568017, -8.5444632, 8.6015549
3: -7.1426382, 2.4905334, -7.2934699, 2.5502114, -9.6928501, 9.7840033
4: -3.3797951, 2.8625700, -3.4804327, 2.9168844, -6.2966795, 6.3430028

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0719180, upper bound: 3.0719180
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0719180, upper bound: 3.0719180
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.7970945, 1.1366076, -1.3068956, 1.5748544, -2.3719487, 2.4435031
1: -9.1047134, 2.0115535, -11.5171356, 3.1241891, -12.2289028, 13.5286894
2: -3.4911079, 2.4991722, -5.1195836, 3.4610801, -6.9521880, 7.6187544
3: -5.3678417, 1.8765234, -7.2925711, 2.5727680, -7.9406080, 9.1690922
4: -2.2343647, 2.2230315, -3.4549942, 3.0060148, -5.2403793, 5.6780248

Time for backsubstitution: 1.44 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.71 + 418.47 = 421.18 seconds
