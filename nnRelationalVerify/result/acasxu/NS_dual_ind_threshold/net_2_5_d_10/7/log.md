## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 655.396249165482


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-305.5349121, 611.6073608, -305.5349121, 611.6073608, -917.1422729, 917.1422729)
1: (-107.4886780, 220.3997650, -107.4886780, 220.3997650, -327.8883362, 327.8883362)
2: (-66.6060410, 226.1742706, -66.6060410, 226.1742706, -292.7802734, 292.7802734)
3: (-134.5018005, 264.9532471, -134.5018005, 264.9532471, -399.4550476, 399.4550476)
4: (-73.2891235, 221.7262726, -73.2891235, 221.7262726, -295.0153809, 295.0153809)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.21 + 2.04 = 4.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -655.4355753, upper bound: 655.4355753

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4354422, upper bound: 655.4355753
time: 0.83 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4355753, upper bound: 655.4355753
time: 0.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.71 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.71
Output dim: 0, lower bound: -655.4354422, upper bound: 655.4355753
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.71
Output dim: 0, lower bound: -655.4355753, upper bound: 655.4355753

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -290.2778625, 580.8662720, -298.6199646, 597.6947632, -887.9726562, 879.4862061
1: -102.1189194, 209.4102478, -105.0555191, 215.4125366, -317.5314636, 314.4657593
2: -63.3475266, 214.9811554, -65.1285248, 221.1005096, -284.4479980, 280.1096802
3: -127.7948303, 251.7710419, -131.4608765, 258.9732056, -386.7678528, 383.2319031
4: -69.6624374, 210.6902466, -71.6446838, 216.7233734, -286.3858032, 282.3348389

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4353579, upper bound: 655.4352984
time: 0.70 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4351147, upper bound: 655.4352084
time: 0.75 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -327.9538269, 658.1427612, -301.9416199, 604.4362183, -932.3900146, 960.0843506
1: -115.8150711, 237.7415619, -106.2395859, 217.8506775, -333.6657410, 343.9811401
2: -71.7138748, 243.7386017, -65.8412170, 223.5668945, -295.2807617, 309.5798340
3: -144.7678070, 285.6264038, -132.9446259, 261.8687439, -406.6365356, 418.5710449
4: -78.9475021, 238.5701752, -72.4365234, 219.1653748, -298.1128845, 311.0067139

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4352563, upper bound: 655.4343687
time: 0.95 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4345000, upper bound: 655.4345000
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.91 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.91
Output dim: 0, lower bound: -655.4353579, upper bound: 655.4352984
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.91
Output dim: 0, lower bound: -655.4351147, upper bound: 655.4352084
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.91
Output dim: 0, lower bound: -655.4352563, upper bound: 655.4343687
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.91
Output dim: 0, lower bound: -655.4345000, upper bound: 655.4345000

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -286.6577148, 573.3933105, -283.3150940, 566.8265381, -853.4842529, 856.7083740
1: -100.8400192, 206.7576599, -99.6005402, 204.1541595, -304.9941711, 306.3581543
2: -62.5712585, 212.2764893, -61.7884140, 209.6102295, -272.1814575, 274.0649109
3: -126.2253265, 248.5581818, -124.6947327, 245.4279480, -371.6532593, 373.2528076
4: -68.7922134, 208.0755920, -67.9441910, 205.4757233, -274.2679138, 276.0197754

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344727, upper bound: 655.4351006
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4339337, upper bound: 655.4323101
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344905, upper bound: 655.4345267
time: 0.68 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -289.0890198, 578.5159302, -291.5197144, 583.6599121, -872.7489014, 870.0356445
1: -101.7127914, 208.5869598, -102.6246719, 210.4915924, -312.2043762, 311.2116089
2: -63.0944557, 214.1371765, -63.6135559, 216.0484161, -279.1428833, 277.7507324
3: -127.2845383, 250.7699280, -128.4016724, 253.0020905, -380.2866211, 379.1715698
4: -69.3820038, 209.8624573, -69.9716492, 211.7624512, -281.1444397, 279.8341064

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4343557, upper bound: 655.4350506
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335778, upper bound: 655.4322026
time: 0.86 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -325.1237488, 652.5226440, -269.2953796, 539.9754028, -865.0991211, 921.8179321
1: -114.8608322, 235.8113403, -95.1843719, 195.7463074, -310.6071472, 330.9957275
2: -71.1159286, 241.7427368, -58.9758682, 200.7712860, -271.8872070, 300.7185974
3: -143.5721893, 283.2771606, -119.0680618, 235.0877686, -378.6598816, 402.3452148
4: -78.2887573, 236.5829620, -64.8558655, 196.4632111, -274.7519226, 301.4387817

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4343687, upper bound: 655.4343687
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4343687, upper bound: 655.4343687
time: 0.77 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -318.0543213, 638.9046631, -504.4119263, 1019.4734497, -1333.9848633, 1143.3165283
1: -112.3337860, 230.5267792, -178.9838715, 363.2031860, -475.5043640, 409.5106506
2: -69.6141205, 236.4743805, -111.1140518, 373.7732849, -442.2706299, 347.5884399
3: -140.3289337, 277.0871887, -221.3168030, 438.5737915, -577.8096924, 498.4039917
4: -76.6076736, 231.5390625, -122.8455658, 365.4757385, -441.3258667, 354.3845825

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4343209, upper bound: 655.4343234
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4342558, upper bound: 655.4342558
time: 0.85 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.87 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 0, lower bound: -655.4339337, upper bound: 655.4323101
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 0, lower bound: -655.4344905, upper bound: 655.4345267
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 0, lower bound: -655.4343557, upper bound: 655.4350506
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 0, lower bound: -655.4335778, upper bound: 655.4322026
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 0, lower bound: -655.4343687, upper bound: 655.4343687
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 0, lower bound: -655.4343687, upper bound: 655.4343687
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 0, lower bound: -655.4343209, upper bound: 655.4343234
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 0, lower bound: -655.4342558, upper bound: 655.4342558

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -259.6924438, 521.6871948, -266.3328857, 533.0603027, -792.7526855, 788.0200195
1: -91.6422501, 188.1121674, -93.7296219, 192.0832214, -283.7254333, 281.8417969
2: -56.7702179, 193.3166046, -58.1638985, 197.2837372, -254.0539551, 251.4804993
3: -114.5543976, 226.5185394, -117.3709030, 230.8338165, -345.3882141, 343.8894043
4: -62.4854012, 189.4013062, -63.9088554, 193.4838562, -255.9692535, 253.3101654

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4339337, upper bound: 655.4322109
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4339337, upper bound: 655.4323101
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -277.7493896, 555.4141235, -282.3968506, 564.9768066, -842.7261963, 837.8109741
1: -97.6917801, 200.4533386, -99.2765274, 203.5098267, -301.2015991, 299.7298584
2: -60.6500931, 205.7646790, -61.5904694, 208.9439392, -269.5940247, 267.3551331
3: -122.2957993, 240.9502869, -124.2926559, 244.6488342, -366.9446106, 365.2429504
4: -66.6445389, 201.6751862, -67.7237625, 204.8203430, -271.4648743, 269.3989563

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344683, upper bound: 655.4345227
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344683, upper bound: 655.4345267
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -269.9642944, 540.7689819, -285.1411438, 571.0375366, -841.0018311, 825.9100952
1: -95.2614288, 195.8573608, -100.4803925, 206.2497864, -301.5111694, 296.3377380
2: -59.0648804, 200.9522552, -62.2729759, 211.6558685, -270.7207642, 263.2252197
3: -119.2305069, 235.3087006, -125.7184677, 247.8497620, -367.0802612, 361.0271606
4: -64.9344482, 196.7013245, -68.4910278, 207.3746796, -272.3091431, 265.1923523

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4316095, upper bound: 655.4276186
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4220535, upper bound: 655.4257658
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -356.2402954, 720.8172607, -290.8494873, 582.3205566, -938.5607300, 1011.6667480
1: -126.9352951, 260.3699951, -102.3923111, 210.0343628, -336.9696655, 362.7622681
2: -78.6671829, 267.5027771, -63.4714928, 215.5712433, -294.2384338, 330.7158813
3: -157.8762360, 313.2099304, -128.1177826, 252.4535675, -410.3297729, 441.0559387
4: -86.6412506, 261.1962891, -69.8140106, 211.2915192, -297.9327393, 330.8435364

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4284478, upper bound: 655.4226483
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4215267, upper bound: 655.4213589
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -302.3978577, 607.4686890, -269.2953796, 539.9754028, -842.3732910, 876.7639771
1: -107.1737442, 220.2392426, -95.1843719, 195.7463074, -302.9200439, 315.4236145
2: -66.3173981, 225.6740723, -58.9758682, 200.7712860, -267.0886841, 284.6499023
3: -133.9076385, 264.3871765, -119.0680618, 235.0877686, -368.9953613, 383.4551392
4: -72.9900665, 220.5296936, -64.8558655, 196.4632111, -269.4532166, 285.3854980

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4350736, upper bound: 655.4341356
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4347448, upper bound: 655.4342760
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -529.3274536, 1070.0690918, -269.2953796, 539.9754028, -1069.3028564, 1333.7772217
1: -187.9961243, 381.2335510, -95.1843719, 195.7463074, -383.7424316, 476.0829468
2: -116.6483078, 392.2948608, -58.9758682, 200.7712860, -317.4195862, 449.7603455
3: -232.3466034, 460.2007751, -119.0680618, 235.0877686, -467.4343262, 577.7520752
4: -128.9906921, 383.3810730, -64.8558655, 196.4632111, -325.4538269, 447.1996460

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4350736, upper bound: 655.4341356
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4347447, upper bound: 655.4342760
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -303.3558960, 609.5346069, -499.7882996, 1009.9149170, -1309.5889893, 1109.3228760
1: -107.1254959, 219.7795563, -177.3535767, 359.8207092, -466.9045715, 397.1331177
2: -66.4259033, 225.5301819, -110.1193161, 370.3351440, -435.6063232, 335.6494751
3: -133.8872681, 264.1618042, -219.3213806, 434.4536133, -567.2001343, 483.4831848
4: -73.0757523, 220.8374481, -121.7261429, 362.1530457, -434.4494324, 342.5635071

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314553, upper bound: 655.4330201
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335858, upper bound: 655.4335287
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -311.1177368, 625.1932983, -502.8285828, 1016.3916626, -1323.9597168, 1128.0217285
1: -109.9525223, 225.7106171, -178.4442749, 362.1135254, -472.0514526, 404.1547546
2: -68.1283188, 231.5311432, -110.7772369, 372.6578674, -439.6628418, 342.3083801
3: -137.3408661, 271.2440796, -220.6373138, 437.2528992, -573.5034790, 491.8814087
4: -74.9699631, 226.6824951, -122.4720917, 364.3814087, -438.5927124, 349.1546021

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314494, upper bound: 655.4330660
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335845, upper bound: 655.4335845
time: 0.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.76 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4339337, upper bound: 655.4322109
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4339337, upper bound: 655.4323101
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4344683, upper bound: 655.4345227
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4344683, upper bound: 655.4345267
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4316095, upper bound: 655.4276186
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4220535, upper bound: 655.4257658
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4284478, upper bound: 655.4226483
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4215267, upper bound: 655.4213589
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4350736, upper bound: 655.4341356
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4347448, upper bound: 655.4342760
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4350736, upper bound: 655.4341356
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4347447, upper bound: 655.4342760
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4314553, upper bound: 655.4330201
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4335858, upper bound: 655.4335287
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4314494, upper bound: 655.4330660
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -655.4335845, upper bound: 655.4335845

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -259.6924438, 521.6871948, -258.1929626, 516.6452026, -776.3376465, 779.8800049
1: -91.6422501, 188.1121674, -90.8625183, 186.2115021, -277.8536987, 278.9746704
2: -56.7702179, 193.3166046, -56.4228783, 191.3102875, -248.0805054, 249.7394714
3: -114.5543976, 226.5185394, -113.7839127, 223.8020172, -338.3564148, 340.3023987
4: -62.4854012, 189.4013062, -61.9726067, 187.5890350, -250.0744324, 251.3739166

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336302, upper bound: 655.4314096
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337884, upper bound: 655.4319766
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -259.6924438, 521.6871948, -295.7812500, 593.6404419, -853.3328857, 817.4684448
1: -91.6422501, 188.1121674, -104.5208130, 214.3921509, -306.0343933, 292.6329956
2: -56.7702179, 193.3166046, -64.7823792, 219.9487762, -276.7189636, 258.0989075
3: -114.5543976, 226.5185394, -130.7011261, 257.5359497, -372.0903320, 357.2196350
4: -62.4854012, 189.4013062, -71.2387695, 215.3582458, -277.8436584, 260.6400146

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336302, upper bound: 655.4314380
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337884, upper bound: 655.4320831
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -277.7493896, 555.4141235, -255.5835571, 513.1085205, -790.8579102, 810.9976807
1: -97.6917801, 200.4533386, -90.1062698, 184.7986145, -282.4903870, 290.5596008
2: -60.6500931, 205.7646790, -55.8204880, 189.8988953, -250.5489807, 261.5851135
3: -122.2957993, 240.9502869, -112.6917114, 222.4173737, -344.7130737, 353.6419983
4: -66.6445389, 201.6751862, -61.4258423, 186.1421204, -252.7866516, 263.1010132

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344683, upper bound: 655.4345227
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344683, upper bound: 655.4345227
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -277.7493896, 555.4141235, -276.1315613, 552.2820435, -830.0314331, 831.5455933
1: -97.6917801, 200.4533386, -97.0601120, 199.1434631, -296.8352356, 297.5134583
2: -60.6500931, 205.7646790, -60.2418900, 204.4110718, -265.0611572, 266.0065308
3: -122.2957993, 240.9502869, -121.5433044, 239.3658600, -361.6615906, 362.4935608
4: -66.6445389, 201.6751862, -66.2188263, 200.3529510, -266.9974976, 267.8940125

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344683, upper bound: 655.4345267
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344683, upper bound: 655.4345267
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -266.4569092, 533.4516602, -277.6121216, 555.3815308, -821.8384399, 811.0636597
1: -94.0096359, 193.2651672, -97.7928162, 200.6754303, -294.6850586, 291.0579834
2: -58.2993622, 198.2867279, -60.6289597, 205.9233246, -264.2226868, 258.9156799
3: -117.6834488, 232.1845398, -122.3988647, 241.1322021, -358.8156433, 354.5834045
4: -64.0839767, 194.1075745, -66.6644211, 201.8029327, -265.8868713, 260.7719116

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4295882, upper bound: 655.4264997
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4303507, upper bound: 655.4272996
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -259.0039368, 519.2797852, -297.3912048, 593.6250610, -852.6290283, 816.6710205
1: -91.4404526, 188.2292023, -104.4433441, 215.2259827, -306.6664429, 292.6725464
2: -56.7057228, 193.1205902, -64.7230606, 220.4060059, -277.1117249, 257.8436584
3: -114.4800034, 226.1484222, -130.9234314, 258.6744080, -373.1544189, 357.0717773
4: -62.3508797, 188.9780731, -71.3086472, 215.8351898, -278.1860657, 260.2866821

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4218883, upper bound: 655.4251417
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4220238, upper bound: 655.4257276
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -352.4278259, 713.0591431, -283.4198303, 566.8707886, -919.2985229, 996.4788818
1: -125.5830307, 257.5871887, -99.7406464, 204.5336914, -330.1167297, 357.3278198
2: -77.8392105, 264.6477661, -61.8482666, 209.9140472, -287.7532654, 326.2064819
3: -156.2056885, 309.8604431, -124.8408356, 245.8244019, -402.0300598, 434.3874207
4: -85.7249985, 258.4094238, -68.0106964, 205.7917023, -291.5166626, 326.2315369

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4262484, upper bound: 655.4221659
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4263569, upper bound: 655.4222467
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -345.8102722, 700.5138550, -302.6492615, 604.0103149, -949.8205566, 1003.1630859
1: -123.3394699, 253.1413574, -106.2059021, 218.7058411, -342.0453186, 359.3472595
2: -76.4294510, 260.0768127, -65.8273315, 224.0113983, -300.4407959, 325.6659241
3: -153.3789368, 304.5253906, -133.1304321, 262.9016724, -416.2806091, 437.3791199
4: -84.1976013, 253.8616333, -72.5260544, 219.4449615, -303.6425781, 326.2205505

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4214019, upper bound: 655.4212405
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4214965, upper bound: 655.4213238
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -297.6435852, 597.9443359, -252.1316223, 506.1053162, -803.7489014, 850.0759277
1: -105.5604401, 217.0115967, -89.3914490, 184.3116608, -289.8721008, 306.4030457
2: -65.3175125, 222.3313446, -55.3641891, 188.9115295, -254.2290192, 277.6955261
3: -131.8825226, 260.4758911, -111.8365631, 221.1707916, -353.0532532, 372.3124390
4: -71.8818207, 217.1842041, -60.8589401, 184.5991669, -256.4809265, 278.0431519

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4342275, upper bound: 655.4337101
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4342297, upper bound: 655.4338419
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -297.1026001, 597.4022827, -335.9171753, 681.2936401, -978.3961792, 933.3193970
1: -105.3815308, 216.6777191, -120.1913452, 246.7933960, -352.1749268, 336.8690186
2: -65.1793365, 222.0061798, -74.3978806, 253.4635468, -318.2824097, 296.4040527
3: -131.6803436, 260.0984802, -149.3685455, 296.7457275, -427.9720764, 409.4670410
4: -71.7520981, 216.9258881, -81.9616394, 247.1880493, -318.6945190, 298.8875122

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4339632, upper bound: 655.4338252
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4339572, upper bound: 655.4339572
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -524.0663452, 1059.0782471, -252.1316223, 506.1053162, -1030.1716309, 1305.3873291
1: -186.1481781, 377.5452576, -89.3914490, 184.3116608, -370.4597778, 466.5847168
2: -115.5099564, 388.4946594, -55.3641891, 188.9115295, -304.4214783, 442.3179932
3: -230.0730286, 455.7357178, -111.8365631, 221.1707916, -451.2438049, 566.0266724
4: -127.7331467, 379.6213074, -60.8589401, 184.5991669, -312.3322449, 439.4255066

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4339773, upper bound: 655.4329455
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4346042, upper bound: 655.4340205
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -523.8265991, 1059.8011475, -335.9171753, 681.2936401, -1205.1202393, 1390.1284180
1: -186.1629944, 377.4364319, -120.1913452, 246.7933960, -432.3780212, 496.8702698
2: -115.5133514, 388.4007263, -74.3978806, 253.4635468, -368.4304810, 461.2114563
3: -230.0862274, 455.6310730, -149.3685455, 296.7457275, -526.2280884, 603.3699951
4: -127.7154617, 379.5982056, -81.9616394, 247.1880493, -374.4746094, 460.4455872

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4339931, upper bound: 655.4331697
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4346731, upper bound: 655.4341887
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -285.6856079, 574.2793579, -474.7120972, 962.2014771, -1242.4691162, 1048.9914551
1: -101.0140610, 207.1313019, -168.9553680, 342.9244080, -443.6326904, 376.0596619
2: -62.6537933, 212.6440582, -104.6939545, 353.1117554, -414.4196777, 317.3379517
3: -126.2121506, 248.9379272, -208.6060181, 414.2290039, -539.0914307, 457.5439453
4: -68.8708572, 208.2776947, -115.8499527, 345.2243347, -413.1220703, 324.1276245

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314220, upper bound: 655.4330201
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314220, upper bound: 655.4329625
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -302.4302673, 607.6691284, -488.9198303, 988.1112061, -1287.0415039, 1096.5889893
1: -106.7991714, 219.1257782, -173.4952087, 352.1690369, -458.9387512, 392.6209412
2: -66.2260437, 224.8560028, -107.7676849, 362.4263916, -427.5332031, 332.6236572
3: -133.4803314, 263.3713074, -214.5357056, 425.2125244, -557.5629272, 477.9070129
4: -72.8527679, 220.1750336, -119.0966721, 354.3906860, -426.4900513, 339.2716675

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335675, upper bound: 655.4334392
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4335675, upper bound: 655.4335287
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -293.9762573, 591.1132202, -478.1459656, 969.5839844, -1258.3690186, 1069.2591553
1: -104.0486755, 213.5223999, -170.1910706, 345.5486145, -449.3359985, 383.7134705
2: -64.4772034, 219.1140900, -105.4385834, 355.7966309, -418.9673767, 324.5526733
3: -129.9184723, 256.5764771, -210.1064758, 417.5060425, -546.1286011, 466.6828918
4: -70.9090500, 214.5668182, -116.7071915, 347.7515259, -417.7438965, 331.2739563

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4310239, upper bound: 655.4318126
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314194, upper bound: 655.4329464
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -310.1604919, 623.2723389, -492.2148132, 995.1294556, -1301.9299316, 1115.4870605
1: -109.6158218, 225.0336456, -174.6799774, 354.6645508, -464.2794495, 399.7136230
2: -67.9222260, 230.8332062, -108.4805679, 364.9508362, -431.7872009, 339.3137817
3: -136.9186707, 270.4267883, -215.9624329, 428.2504578, -564.0926514, 486.3891602
4: -74.7392349, 225.9956665, -119.9042206, 356.8093262, -430.8182068, 345.8998413

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330659, upper bound: 655.4314494
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330659, upper bound: 655.4335845
time: 0.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.33 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4336302, upper bound: 655.4314096
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4337884, upper bound: 655.4319766
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4336302, upper bound: 655.4314380
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4337884, upper bound: 655.4320831
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4344683, upper bound: 655.4345227
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4344683, upper bound: 655.4345227
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4344683, upper bound: 655.4345267
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4344683, upper bound: 655.4345267
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4295882, upper bound: 655.4264997
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4303507, upper bound: 655.4272996
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4218883, upper bound: 655.4251417
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4220238, upper bound: 655.4257276
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4262484, upper bound: 655.4221659
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4263569, upper bound: 655.4222467
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4214019, upper bound: 655.4212405
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4214965, upper bound: 655.4213238
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4342275, upper bound: 655.4337101
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4342297, upper bound: 655.4338419
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4339632, upper bound: 655.4338252
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4339572, upper bound: 655.4339572
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4339773, upper bound: 655.4329455
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4346042, upper bound: 655.4340205
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4339931, upper bound: 655.4331697
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4346731, upper bound: 655.4341887
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4314220, upper bound: 655.4330201
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4314220, upper bound: 655.4329625
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4335675, upper bound: 655.4334392
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4335675, upper bound: 655.4335287
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4310239, upper bound: 655.4318126
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4314194, upper bound: 655.4329464
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4330659, upper bound: 655.4314494
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -655.4330659, upper bound: 655.4335845

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -250.7117615, 504.0198364, -252.4535217, 505.3434448, -756.0551758, 756.4733887
1: -88.5698471, 181.8745575, -88.9007034, 182.2298279, -270.7996826, 270.7752686
2: -54.8863373, 186.9442902, -55.2213440, 187.2377777, -242.1241150, 242.1656342
3: -110.7011414, 219.0093231, -111.3324966, 218.9900513, -329.6911621, 330.3417969
4: -60.3859711, 183.1042938, -60.6316185, 183.5771332, -243.9630737, 243.7358856

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314060
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314096
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -259.9921570, 523.7673950, -255.8327332, 511.9362793, -771.9284668, 779.6000977
1: -91.9585419, 188.4074554, -90.0365906, 184.5267944, -276.4852600, 278.4440308
2: -56.9697838, 193.8698120, -55.9198151, 189.5803986, -246.5501862, 249.7896271
3: -114.9001160, 226.8862000, -112.7628555, 221.7793274, -336.6794434, 339.6490173
4: -62.6757736, 190.0154724, -61.4088173, 185.8987122, -248.5744781, 251.4242554

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337588, upper bound: 655.4319694
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337588, upper bound: 655.4319766
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -250.7117615, 504.0198364, -290.4084167, 582.9522705, -833.6639404, 794.4281616
1: -88.5698471, 181.8745575, -102.6694870, 210.5962677, -299.1661072, 284.5440369
2: -54.8863373, 186.9442902, -63.6489296, 216.0660248, -270.9523621, 250.5932159
3: -110.7011414, 219.0093231, -128.3909760, 252.9638519, -363.6649780, 347.4002991
4: -60.3859711, 183.1042938, -69.9755554, 211.5395508, -271.9254761, 253.0798187

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314353
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314380
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -259.9921570, 523.7673950, -291.4009094, 584.7612915, -844.7534180, 815.1683350
1: -91.9585419, 188.4074554, -102.9776535, 211.2917480, -303.2501831, 291.3851013
2: -56.9697838, 193.8698120, -63.8436508, 216.7581940, -273.7279663, 257.7134399
3: -114.9001160, 226.8862000, -128.8030396, 253.8180084, -368.7180786, 355.6891785
4: -62.6757736, 190.0154724, -70.1881409, 212.2339020, -274.9096680, 260.2036133

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337588, upper bound: 655.4320759
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337588, upper bound: 655.4320831
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -268.0839233, 536.0001831, -255.5835571, 513.1085205, -781.1923218, 791.5837402
1: -94.2226868, 193.3194122, -90.1062698, 184.7986145, -279.0213013, 283.4256897
2: -58.5218658, 198.4834290, -55.8204880, 189.8988953, -248.4207611, 254.3039246
3: -118.0015945, 232.3802185, -112.6917114, 222.4173737, -340.4189453, 345.0719299
4: -64.3036652, 194.5142517, -61.4258423, 186.1421204, -250.4457855, 255.9400940

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4345017, upper bound: 655.4344808
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4345017, upper bound: 655.4345636
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -275.0977173, 550.5734253, -255.5835571, 513.1085205, -788.2059937, 806.1569824
1: -96.8359985, 198.7898865, -90.1062698, 184.7986145, -281.6346130, 288.8961487
2: -60.0969429, 204.0411530, -55.8204880, 189.8988953, -249.9958344, 259.8616333
3: -121.1697769, 238.9312134, -112.6917114, 222.4173737, -343.5871277, 351.6228638
4: -66.0479584, 199.9306488, -61.4258423, 186.1421204, -252.1900787, 261.3564758

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4345017, upper bound: 655.4344808
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4345017, upper bound: 655.4345636
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -268.0839233, 536.0001831, -276.1315613, 552.2820435, -820.3658447, 812.1316528
1: -94.2226868, 193.3194122, -97.0601120, 199.1434631, -293.3661194, 290.3795166
2: -58.5218658, 198.4834290, -60.2418900, 204.4110718, -262.9329224, 258.7253113
3: -118.0015945, 232.3802185, -121.5433044, 239.3658600, -357.3674011, 353.9234619
4: -64.3036652, 194.5142517, -66.2188263, 200.3529510, -264.6566162, 260.7329712

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344905, upper bound: 655.4344781
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344905, upper bound: 655.4345267
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -275.0977173, 550.5734253, -276.1315613, 552.2820435, -827.3796387, 826.7048950
1: -96.8359985, 198.7898865, -97.0601120, 199.1434631, -295.9794617, 295.8499756
2: -60.0969429, 204.0411530, -60.2418900, 204.4110718, -264.5080261, 264.2830505
3: -121.1697769, 238.9312134, -121.5433044, 239.3658600, -360.5356140, 360.4743958
4: -66.0479584, 199.9306488, -66.2188263, 200.3529510, -266.4009094, 266.1494446

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344905, upper bound: 655.4344781
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344905, upper bound: 655.4345267
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -260.6073608, 521.6282959, -277.6121216, 555.3815308, -815.9888916, 799.2402954
1: -91.9638290, 188.9760895, -97.7928162, 200.6754303, -292.6392517, 286.7689209
2: -57.0453720, 193.9245911, -60.6289597, 205.9233246, -262.9686890, 254.5535583
3: -115.1187439, 227.0241241, -122.3988647, 241.1322021, -356.2509460, 349.4229736
4: -62.6853065, 189.8645325, -66.6644211, 201.8029327, -264.4881592, 256.5289001

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4295882, upper bound: 655.4264996
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4295882, upper bound: 655.4264997
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -258.1728210, 519.4817505, -270.1844788, 541.4360962, -799.6088867, 789.6661987
1: -91.2742691, 187.7443390, -95.2504807, 195.5085754, -286.7828064, 282.9947510
2: -56.3758926, 192.7545166, -59.0166245, 200.6598969, -257.0357666, 251.7711487
3: -114.0997238, 225.6557465, -119.1741638, 234.9661255, -349.0658264, 344.8298950
4: -62.1755714, 188.7087097, -64.9265366, 196.6236267, -258.7991333, 253.6352539

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4303507, upper bound: 655.4272996
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4303507, upper bound: 655.4272996
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -253.0547791, 507.2885437, -297.3912048, 593.6250610, -846.6798096, 804.6797485
1: -89.3622437, 183.8726501, -104.4433441, 215.2259827, -304.5882263, 288.3159790
2: -55.4284286, 188.6861572, -64.7230606, 220.4060059, -275.8344421, 253.4092102
3: -111.8685608, 220.9114227, -130.9234314, 258.6744080, -370.5429382, 351.8347473
4: -60.9292107, 184.6558685, -71.3086472, 215.8351898, -276.7644043, 255.9645081

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4218883, upper bound: 655.4251417
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4218883, upper bound: 655.4251417
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -249.8941498, 504.0943909, -289.1388245, 578.1423340, -828.0364380, 793.2332153
1: -88.4773712, 182.2526398, -101.6470184, 209.5079041, -297.9852905, 283.8995972
2: -54.6191101, 187.1459045, -62.9506340, 214.6219635, -269.2409973, 250.0965424
3: -110.5838242, 219.0462036, -127.3721161, 251.8784943, -362.4623108, 346.4183350
4: -60.2740059, 183.1118927, -69.3914108, 210.1195831, -270.3935547, 252.5032959

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4220238, upper bound: 655.4257276
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4220238, upper bound: 655.4257276
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -346.5928040, 701.4409180, -283.4198303, 566.8707886, -913.4634399, 984.8606567
1: -123.5488129, 253.3388672, -99.7406464, 204.5336914, -328.0825195, 353.0794678
2: -76.5876236, 260.2962646, -61.8482666, 209.9140472, -286.5016174, 321.8478088
3: -153.6601868, 304.6977844, -124.8408356, 245.8244019, -399.4845276, 429.2207947
4: -84.3292999, 254.1936493, -68.0106964, 205.7917023, -290.1209717, 321.9790344

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4253752, upper bound: 655.4219384
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4253752, upper bound: 655.4221659
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -343.2495117, 697.5584717, -276.0058289, 552.9375610, -896.1870728, 973.5642090
1: -122.4884491, 251.5253296, -97.2016525, 199.3725128, -321.8609619, 348.7269287
2: -75.7326660, 258.5316467, -60.2372322, 204.6555939, -280.3882446, 318.5284424
3: -152.2327576, 302.6506348, -121.6209106, 239.6630707, -391.8958130, 424.0410767
4: -83.6068420, 252.4041290, -66.2750244, 200.6190948, -284.2258911, 318.5269165

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4254888, upper bound: 655.4220163
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4254888, upper bound: 655.4222467
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -339.9021301, 688.6079712, -302.6492615, 604.0103149, -943.9124756, 991.2571411
1: -121.2832718, 248.8129272, -106.2059021, 218.7058411, -339.9891052, 355.0188293
2: -75.1621323, 255.6735840, -65.8273315, 224.0113983, -299.1735229, 321.2554932
3: -150.7950439, 299.3075562, -133.1304321, 262.9016724, -413.6967163, 432.1576233
4: -82.7853622, 249.5577393, -72.5260544, 219.4449615, -302.2303162, 321.9094543

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4212249, upper bound: 655.4212363
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4212250, upper bound: 655.4212405
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -335.9059448, 683.9311523, -294.4112854, 588.5474243, -924.4533691, 978.3424072
1: -120.0460281, 246.6592407, -103.4121017, 212.9941559, -333.0401917, 350.0713501
2: -74.1772232, 253.5373535, -64.0571060, 218.2346649, -292.4118652, 317.4059448
3: -149.1345978, 296.8117371, -129.5849609, 256.1138000, -405.2484131, 426.2055359
4: -81.9281158, 247.4117889, -70.6109085, 213.7379913, -295.6660461, 317.8926086

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4213194, upper bound: 655.4213194
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4213194, upper bound: 655.4213238
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -278.8540344, 562.4674683, -236.6553497, 475.0945435, -753.9484863, 799.1228027
1: -99.0830688, 203.9055939, -84.0158615, 173.3080902, -272.3911743, 287.9214478
2: -61.2383156, 209.1721954, -52.0465317, 177.6665039, -238.9048157, 261.2187195
3: -123.6131134, 245.4124603, -105.0914841, 207.9212646, -331.5343628, 350.5039062
4: -67.5029755, 204.1720734, -57.1653252, 173.6261902, -241.1291351, 261.3373718

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337314, upper bound: 655.4312671
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337314, upper bound: 655.4312603
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -289.5789795, 581.7265625, -251.1799774, 504.2016296, -793.7806396, 832.9064941
1: -102.7070160, 211.1864929, -89.0555191, 183.6399689, -286.3469849, 300.2420044
2: -63.5708542, 216.3475189, -55.1591644, 188.2191162, -251.7899780, 271.5066833
3: -128.2966003, 253.5041962, -111.4174042, 220.3637238, -348.6602173, 364.9216003
4: -69.9294891, 211.3253632, -60.6298904, 183.9182892, -253.8477783, 271.9552612

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4342237, upper bound: 655.4338323
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4342237, upper bound: 655.4338313
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -277.3909912, 560.1605225, -320.1782532, 650.1096802, -926.4685669, 880.3386841
1: -98.6105804, 202.8965607, -114.7450180, 235.5496521, -334.0818787, 317.6415710
2: -60.9120674, 208.1604614, -71.0439682, 242.0077362, -302.3692627, 279.2043762
3: -123.0207138, 244.2210388, -142.5771790, 283.2449341, -405.5978088, 386.7982178
4: -67.1606750, 203.2504730, -78.2375870, 236.0308533, -302.7906189, 281.4880676

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334405, upper bound: 655.4314075
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334550, upper bound: 655.4313997
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -288.7715759, 580.6829834, -334.8986511, 679.2579346, -968.0295410, 915.5816650
1: -102.4424973, 210.6770782, -119.8334503, 246.0714264, -348.5138855, 330.5105286
2: -63.3766060, 215.8399811, -74.1791992, 252.7214813, -315.7516785, 290.0191040
3: -127.9802017, 252.9136353, -148.9217834, 295.8788757, -423.4161072, 401.8354187
4: -69.7376709, 210.8833923, -81.7174454, 246.4604034, -315.9620361, 292.6008301

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4339400, upper bound: 655.4339534
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4339530, upper bound: 655.4339530
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -514.7603760, 1039.6630859, -247.3466187, 496.6242065, -1011.3845825, 1280.2995605
1: -182.8523407, 370.8261414, -87.7504272, 180.9738770, -363.8262329, 458.1746521
2: -113.5228653, 381.6095276, -54.3503113, 185.4841919, -299.0070496, 434.3372192
3: -226.0461426, 447.6200256, -109.7698593, 217.1414337, -443.1875610, 555.7935791
4: -125.5080338, 372.9020081, -59.7316780, 181.2164917, -306.7245178, 431.5205688

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4339773, upper bound: 655.4327661
time: 1.99 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4339256, upper bound: 655.4329268
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -565.9715576, 1141.3312988, -245.9131927, 493.6878357, -1059.6593018, 1376.7230225
1: -201.3063049, 408.7546082, -87.2468719, 179.9405823, -380.9995117, 494.6190186
2: -124.7864990, 420.0739746, -54.0480194, 184.4292145, -309.2156982, 471.3182068
3: -248.1503143, 492.6690979, -109.1564560, 215.8941345, -464.0443726, 599.0501099
4: -137.8150482, 409.6233826, -59.3954163, 180.1850739, -318.0000916, 467.0101929

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4345243, upper bound: 655.4338319
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344895, upper bound: 655.4337627
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -514.8408203, 1041.0559082, -331.0855408, 671.7661743, -1186.6069336, 1365.6501465
1: -182.9667816, 370.9125977, -118.5407104, 243.4312134, -425.6731262, 488.6339111
2: -113.5847092, 381.7230225, -73.3786163, 250.0191345, -363.0399475, 453.4309692
3: -226.1676483, 447.7606201, -147.2972260, 292.7017212, -518.1890259, 593.3693237
4: -125.5614777, 373.0827332, -80.8343506, 243.7910767, -368.8964539, 452.7431946

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4329234, upper bound: 655.4321393
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4330393, upper bound: 655.4327150
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -565.6580200, 1142.5906982, -328.2948608, 666.2414551, -1231.6323242, 1460.5230713
1: -201.3784332, 408.7890625, -117.5900269, 241.4287262, -441.5963440, 524.5861206
2: -124.7878799, 420.1358948, -72.7886810, 247.9710693, -372.0708618, 490.0779419
3: -248.1845093, 492.7105713, -146.1031189, 290.2799377, -537.5337524, 635.9470825
4: -137.8176117, 409.7234802, -80.1706085, 241.7931061, -379.0718384, 487.8259583

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4333802, upper bound: 655.4332290
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4334831, upper bound: 655.4332016
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -270.4864502, 543.3883667, -471.8477783, 956.8577881, -1221.8859863, 1015.2361450
1: -95.9318924, 197.0041504, -167.9765778, 340.9877930, -436.5005798, 364.9639282
2: -59.4335327, 202.0098724, -104.0860367, 351.1314392, -409.2143860, 306.0958862
3: -119.9168243, 236.4604034, -207.3636627, 411.9266357, -530.3475342, 443.8240356
4: -65.3316345, 197.4482880, -115.1774521, 343.2669678, -407.6164246, 312.6256104

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314220, upper bound: 655.4330201
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314220, upper bound: 655.4330201
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -481.4034729, 973.1845093, -477.7546387, 967.8712769, -1443.7532959, 1444.1711426
1: -171.3598328, 346.9895020, -169.9951630, 344.9796448, -515.1295166, 515.6743774
2: -106.1910400, 357.3456116, -105.3400040, 355.2136230, -459.8877258, 460.7852173
3: -211.6448059, 418.2164612, -209.9266052, 416.6727295, -626.6240845, 626.3194580
4: -117.3090591, 349.3204346, -116.5644684, 347.3016357, -463.4340515, 464.4861145

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314220, upper bound: 655.4329626
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314220, upper bound: 655.4329625
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -278.1334534, 560.6514282, -488.2127075, 986.8004761, -1259.9329834, 1048.8638916
1: -98.5013123, 202.1196136, -173.2565765, 351.7056885, -449.8815308, 375.3761597
2: -61.0057678, 207.5834503, -107.6177521, 361.9494629, -421.5967407, 315.2012024
3: -122.9176788, 243.3132629, -214.2312775, 424.6560364, -546.2230225, 457.5444946
4: -67.1480560, 203.2253418, -118.9312286, 353.9166870, -420.1310425, 322.1565552

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314505, upper bound: 655.4334360
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314505, upper bound: 655.4334392
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -296.2727661, 595.1821289, -488.9147644, 988.1017456, -1280.9162598, 1084.0969238
1: -104.6299362, 214.8070679, -173.4935150, 352.1658020, -456.7738953, 388.3005676
2: -64.8985214, 220.3845520, -107.7666168, 362.4230347, -426.2111816, 328.1511841
3: -130.7668610, 258.1450195, -214.5335693, 425.2085876, -554.8671875, 472.6785889
4: -71.3689041, 215.7715454, -119.0954971, 354.3873291, -425.0090332, 334.8670349

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314505, upper bound: 655.4335262
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314505, upper bound: 655.4335262
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -278.9030457, 560.9597168, -474.5311584, 962.3864136, -1235.8590088, 1035.4907227
1: -98.8898697, 202.9812775, -168.9452209, 343.0135193, -441.5264587, 371.9265137
2: -61.2855339, 208.2981873, -104.6790237, 353.1978455, -413.1615295, 312.9771729
3: -123.4028549, 243.8168335, -208.5663605, 414.4648438, -536.5633545, 452.3831482
4: -67.3674927, 203.9136047, -115.8657684, 345.2137146, -411.6435242, 319.7793579

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4310239, upper bound: 655.4318126
time: 1.21 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4310239, upper bound: 655.4318126
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -333.8579712, 673.6181641, -465.8579712, 946.2807007, -1275.0472412, 1139.0452881
1: -118.9899826, 244.3457489, -166.0346985, 336.9946594, -455.2920532, 409.5789795
2: -73.6993256, 250.3640137, -102.8039017, 347.1134033, -419.4085999, 352.5567322
3: -147.8420715, 293.1477356, -204.8353271, 407.2175293, -553.5533447, 497.4804993
4: -80.9311066, 244.1336517, -113.8165817, 339.2563782, -419.2022095, 357.4527588

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314194, upper bound: 655.4329464
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4314194, upper bound: 655.4329464
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -290.8445129, 587.4707642, -491.6342163, 994.0528564, -1280.0212402, 1079.1046143
1: -103.1763000, 211.8757935, -174.4839935, 354.2838135, -457.2047424, 386.3598022
2: -63.7978058, 217.5765686, -108.3576202, 364.5590820, -427.0234375, 325.9341431
3: -128.6190033, 255.1755066, -215.7124481, 427.7936707, -555.1234131, 470.8879089
4: -70.3013916, 212.8874512, -119.7685623, 356.4200745, -425.8100281, 332.6560059

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309719, upper bound: 655.4314494
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309719, upper bound: 655.4314494
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -302.8418579, 608.6545410, -492.0675659, 994.8565063, -1294.3834229, 1100.7219238
1: -107.0545044, 219.8508453, -174.6303101, 354.5679932, -461.6224670, 394.4811096
2: -66.3478317, 225.4879303, -108.4494171, 364.8515320, -430.1239929, 333.9373474
3: -133.6790924, 264.1714172, -215.8990479, 428.1346130, -560.7601318, 480.0704651
4: -72.9787369, 220.7287140, -119.8698425, 356.7106934, -428.9659729, 340.5985718

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309719, upper bound: 655.4335824
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4309719, upper bound: 655.4335845
time: 0.86 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.64 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314060
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314096
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4337588, upper bound: 655.4319694
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4337588, upper bound: 655.4319766
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314353
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314380
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4337588, upper bound: 655.4320759
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4337588, upper bound: 655.4320831
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4345017, upper bound: 655.4344808
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4345017, upper bound: 655.4345636
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4345017, upper bound: 655.4344808
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4345017, upper bound: 655.4345636
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4344905, upper bound: 655.4344781
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4344905, upper bound: 655.4345267
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4344905, upper bound: 655.4344781
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4344905, upper bound: 655.4345267
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4295882, upper bound: 655.4264996
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4295882, upper bound: 655.4264997
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4303507, upper bound: 655.4272996
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4303507, upper bound: 655.4272996
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4218883, upper bound: 655.4251417
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4218883, upper bound: 655.4251417
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4220238, upper bound: 655.4257276
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4220238, upper bound: 655.4257276
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4253752, upper bound: 655.4219384
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4253752, upper bound: 655.4221659
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4254888, upper bound: 655.4220163
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4254888, upper bound: 655.4222467
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4212249, upper bound: 655.4212363
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4212250, upper bound: 655.4212405
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4213194, upper bound: 655.4213194
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4213194, upper bound: 655.4213238
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4337314, upper bound: 655.4312671
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4337314, upper bound: 655.4312603
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4342237, upper bound: 655.4338323
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4342237, upper bound: 655.4338313
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4334405, upper bound: 655.4314075
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4334550, upper bound: 655.4313997
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4339400, upper bound: 655.4339534
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4339530, upper bound: 655.4339530
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4339773, upper bound: 655.4327661
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4339256, upper bound: 655.4329268
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4345243, upper bound: 655.4338319
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4344895, upper bound: 655.4337627
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4329234, upper bound: 655.4321393
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4330393, upper bound: 655.4327150
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4333802, upper bound: 655.4332290
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4334831, upper bound: 655.4332016
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4314220, upper bound: 655.4330201
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4314220, upper bound: 655.4330201
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4314220, upper bound: 655.4329626
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4314220, upper bound: 655.4329625
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4314505, upper bound: 655.4334360
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4314505, upper bound: 655.4334392
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4314505, upper bound: 655.4335262
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4314505, upper bound: 655.4335262
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4310239, upper bound: 655.4318126
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4310239, upper bound: 655.4318126
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4314194, upper bound: 655.4329464
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4314194, upper bound: 655.4329464
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4309719, upper bound: 655.4314494
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4309719, upper bound: 655.4314494
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4309719, upper bound: 655.4335824
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -655.4309719, upper bound: 655.4335845

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -250.7117615, 504.0198364, -241.2794800, 484.3848572, -735.0964966, 745.2993164
1: -88.5698471, 181.8745575, -85.0983887, 174.5894470, -263.1592712, 266.9729004
2: -54.8863373, 186.9442902, -52.7780037, 179.4895172, -234.3758545, 239.7222900
3: -110.7011414, 219.0093231, -106.4319000, 210.1908264, -320.8919067, 325.4411621
4: -60.3859711, 183.1042938, -58.0396271, 175.8732605, -236.2592010, 241.1439209

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314060
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314060
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -250.7117615, 504.0198364, -261.9691772, 523.9368286, -774.6484985, 765.9889526
1: -88.5698471, 181.8745575, -92.1367722, 189.0723724, -277.6422119, 274.0113220
2: -54.8863373, 186.9442902, -57.2419891, 194.1416168, -249.0279541, 244.1862793
3: -110.7011414, 219.0093231, -115.3921738, 227.2643280, -337.9653625, 334.4014893
4: -60.3859711, 183.1042938, -62.8755798, 190.2403870, -250.6263428, 245.9798584

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314096
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314096
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -259.9921570, 523.7673950, -245.4431152, 492.6068420, -752.5989990, 769.2104492
1: -91.9585419, 188.4074554, -86.5141220, 177.4145355, -269.3730774, 274.9215698
2: -56.9697838, 193.8698120, -53.6473274, 182.3915100, -239.3612976, 247.5171356
3: -114.9001160, 226.8862000, -108.2024689, 213.6010895, -328.5011902, 335.0886536
4: -62.6757736, 190.0154724, -59.0064316, 178.7474976, -241.4232635, 249.0219116

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337588, upper bound: 655.4319694
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337587, upper bound: 655.4319694
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -259.9921570, 523.7673950, -265.5321655, 530.8790283, -790.8710938, 789.2994995
1: -91.9585419, 188.4074554, -93.3342819, 191.5102844, -283.4687805, 281.7417297
2: -56.9697838, 193.8698120, -57.9783745, 196.6219177, -253.5917053, 251.8481903
3: -114.9001160, 226.8862000, -116.9055176, 230.2131805, -345.1132507, 343.7917175
4: -62.6757736, 190.0154724, -63.6952286, 192.6915131, -255.3672791, 253.7106781

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337588, upper bound: 655.4319766
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337588, upper bound: 655.4319763
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -250.7117615, 504.0198364, -282.1417236, 567.6403198, -818.3520508, 786.1614380
1: -88.5698471, 181.8745575, -99.8493576, 204.9241943, -293.4940186, 281.7239075
2: -54.8863373, 186.9442902, -61.8281670, 210.3522949, -265.2386169, 248.7724609
3: -110.7011414, 219.0093231, -124.7479019, 246.5699921, -357.2711182, 343.7571411
4: -60.3859711, 183.1042938, -68.0626678, 205.8760376, -266.2620239, 251.1669006

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314353
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314353
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -250.7117615, 504.0198364, -300.4144592, 602.8970337, -853.6087646, 804.4342651
1: -88.5698471, 181.8745575, -106.1113358, 217.8954468, -306.4652100, 287.9858704
2: -54.8863373, 186.9442902, -65.7806168, 223.4363556, -278.3226624, 252.7249146
3: -110.7011414, 219.0093231, -132.7370605, 261.7245483, -372.4256897, 351.7463684
4: -60.3859711, 183.1042938, -72.3478317, 218.6714783, -279.0574341, 255.4521179

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314380
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4336022, upper bound: 655.4314380
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -259.9921570, 523.7673950, -282.6988220, 568.6405640, -828.6326904, 806.4660034
1: -91.9585419, 188.4074554, -100.0104523, 205.3210907, -297.2796021, 288.4179077
2: -56.9697838, 193.8698120, -61.9355927, 210.7361603, -267.7059326, 255.8053894
3: -114.9001160, 226.8862000, -124.9784241, 247.0609436, -361.9610596, 351.8646240
4: -62.6757736, 190.0154724, -68.1790237, 206.2705231, -268.9462891, 258.1944885

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337588, upper bound: 655.4320759
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337588, upper bound: 655.4320759
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -259.9921570, 523.7673950, -302.4338989, 606.6727295, -866.6648560, 826.2012939
1: -91.9585419, 188.4074554, -106.7652817, 219.2877045, -311.2462463, 295.1727295
2: -56.9697838, 193.8698120, -66.1899719, 224.8430786, -281.8128662, 260.0597534
3: -114.9001160, 226.8862000, -133.5753326, 263.4194031, -378.3194885, 360.4615479
4: -62.6757736, 190.0154724, -72.8003387, 220.0706024, -282.7463684, 262.8157654

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337587, upper bound: 655.4320831
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337588, upper bound: 655.4320831
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -268.0839233, 536.0001831, -246.8528748, 495.4326782, -763.5166016, 782.8529663
1: -94.2226868, 193.3194122, -87.0134811, 178.4796906, -272.7023926, 280.3328857
2: -58.5218658, 198.4834290, -53.9476013, 183.4640350, -241.9859009, 252.4310303
3: -118.0015945, 232.3802185, -108.8263779, 214.8706970, -332.8722534, 341.2066040
4: -64.3036652, 194.5142517, -59.3444252, 179.7931976, -244.0968475, 253.8586578

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4341239, upper bound: 655.4343208
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4343449, upper bound: 655.4343814
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -268.0839233, 536.0001831, -284.9376526, 573.0709229, -841.1548462, 820.9377441
1: -94.2226868, 193.3194122, -100.7934723, 206.9977722, -301.2204590, 294.1128845
2: -58.5218658, 198.4834290, -62.4114380, 212.4264526, -270.9482727, 260.8948669
3: -118.0015945, 232.3802185, -125.9681549, 249.0574799, -367.0590820, 358.3483582
4: -64.3036652, 194.5142517, -68.7127838, 207.9064636, -272.2101440, 263.2269897

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4341239, upper bound: 655.4343763
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4343449, upper bound: 655.4344917
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -275.0977173, 550.5734253, -246.8528748, 495.4326782, -770.5303955, 797.4261475
1: -96.8359985, 198.7898865, -87.0134811, 178.4796906, -275.3156738, 285.8033752
2: -60.0969429, 204.0411530, -53.9476013, 183.4640350, -243.5609741, 257.9887695
3: -121.1697769, 238.9312134, -108.8263779, 214.8706970, -336.0404663, 347.7575378
4: -66.0479584, 199.9306488, -59.3444252, 179.7931976, -245.8411407, 259.2750549

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4340397, upper bound: 655.4343131
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4337751, upper bound: 655.4343737
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -275.0977173, 550.5734253, -284.9376526, 573.0709229, -848.1685791, 835.5109863
1: -96.8359985, 198.7898865, -100.7934723, 206.9977722, -303.8337708, 299.5833435
2: -60.0969429, 204.0411530, -62.4114380, 212.4264526, -272.5233765, 266.4525757
3: -121.1697769, 238.9312134, -125.9681549, 249.0574799, -370.2272644, 364.8992920
4: -66.0479584, 199.9306488, -68.7127838, 207.9064636, -273.9544067, 268.6434326

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4340397, upper bound: 655.4343465
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4343266, upper bound: 655.4344649
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -268.0839233, 536.0001831, -268.0839233, 536.0001831, -804.0840454, 804.0840454
1: -94.2226868, 193.3194122, -94.2226868, 193.3194122, -287.5421143, 287.5421143
2: -58.5218658, 198.4834290, -58.5218658, 198.4834290, -257.0052795, 257.0053101
3: -118.0015945, 232.3802185, -118.0015945, 232.3802185, -350.3818054, 350.3818054
4: -64.3036652, 194.5142517, -64.3036652, 194.5142517, -258.8179016, 258.8178711

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4332124, upper bound: 655.4340621
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4345116, upper bound: 655.4345117
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -268.0839233, 536.0001831, -305.9209595, 613.8075562, -881.8914795, 841.9211426
1: -94.2226868, 193.3194122, -108.0058136, 221.7746735, -315.9973450, 301.3252258
2: -58.5218658, 198.4834290, -66.9397812, 227.4006500, -285.9225159, 265.4231873
3: -118.0015945, 232.3802185, -135.0958252, 266.3979187, -384.3994751, 367.4760437
4: -64.3036652, 194.5142517, -73.6415100, 222.5709381, -286.8746033, 268.1557312

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4332124, upper bound: 655.4340621
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4345116, upper bound: 655.4345543
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -275.0977173, 550.5734253, -268.0839233, 536.0001831, -811.0977173, 818.6572876
1: -96.8359985, 198.7898865, -94.2226868, 193.3194122, -290.1553955, 293.0125427
2: -60.0969429, 204.0411530, -58.5218658, 198.4834290, -258.5803833, 262.5630188
3: -121.1697769, 238.9312134, -118.0015945, 232.3802185, -353.5499878, 356.9327393
4: -66.0479584, 199.9306488, -64.3036652, 194.5142517, -260.5621643, 264.2343140

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4328330, upper bound: 655.4334738
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344905, upper bound: 655.4344776
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -275.0977173, 550.5734253, -305.9209595, 613.8075562, -888.9052734, 856.4943848
1: -96.8359985, 198.7898865, -108.0058136, 221.7746735, -318.6106567, 306.7957153
2: -60.0969429, 204.0411530, -66.9397812, 227.4006500, -287.4975891, 270.9809265
3: -121.1697769, 238.9312134, -135.0958252, 266.3979187, -387.5676880, 374.0269775
4: -66.0479584, 199.9306488, -73.6415100, 222.5709381, -288.6188965, 273.5721436

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4328330, upper bound: 655.4334738
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4344905, upper bound: 655.4345267
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -260.6073608, 521.6282959, -264.9064026, 530.1860962, -790.7934570, 786.5344849
1: -91.9638290, 188.9760895, -93.5045166, 192.2133331, -284.1771545, 282.4805908
2: -57.0453720, 193.9245911, -57.9522057, 197.1582031, -254.2035828, 251.8768005
3: -115.1187439, 227.0241241, -117.0431671, 230.8492584, -345.9679565, 344.0672913
4: -62.6853065, 189.8645325, -63.7084198, 193.0503998, -255.7357025, 253.5729370

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4295882, upper bound: 655.4264997
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4295882, upper bound: 655.4264997
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -260.6073608, 521.6282959, -348.5483398, 705.2312012, -965.8385620, 870.1766357
1: -91.9638290, 188.9760895, -124.2036057, 254.7516785, -346.7154846, 313.1796875
2: -57.0453720, 193.9245911, -76.9762497, 261.7304993, -318.5063171, 270.9008484
3: -115.1187439, 227.0241241, -154.4916992, 306.4045410, -421.2220459, 381.5158081
4: -62.6853065, 189.8645325, -84.7753143, 255.6130676, -318.1235962, 274.6398010

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4295882, upper bound: 655.4264997
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4295882, upper bound: 655.4264997
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -258.1728210, 519.4817505, -257.4846497, 516.2810669, -774.4538574, 776.9664307
1: -91.2742691, 187.7443390, -90.9645691, 187.0566711, -278.3308411, 278.7088318
2: -56.3758926, 192.7545166, -56.3405342, 191.9039459, -248.2798462, 249.0950317
3: -114.0997238, 225.6557465, -113.8229904, 224.6957703, -338.7954712, 339.4787292
4: -62.1755714, 188.7087097, -61.9722176, 187.8774109, -250.0529785, 250.6809082

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4295882, upper bound: 655.4272996
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4303507, upper bound: 655.4272996
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -258.1728210, 519.4817505, -341.1136475, 691.2788086, -949.4516602, 860.5953369
1: -91.2742691, 187.7443390, -121.6618195, 249.5940094, -340.8681946, 309.4060974
2: -56.3758926, 192.7545166, -75.3701935, 256.4680481, -312.6219177, 268.1246948
3: -114.0997238, 225.6557465, -151.2651825, 300.2493591, -414.1283569, 376.9209290
4: -62.1755714, 188.7087097, -83.0425949, 250.4313660, -312.4573059, 271.7512512

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4303507, upper bound: 655.4272996
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4303507, upper bound: 655.4272996
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -253.0547791, 507.2885437, -284.8892822, 568.9028931, -821.9576416, 792.1777954
1: -89.3622437, 183.8726501, -100.2311401, 206.8742676, -296.2365112, 284.1037903
2: -55.4284286, 188.6861572, -62.0870552, 211.7617035, -267.1901245, 250.7732086
3: -111.8685608, 220.9114227, -125.6578674, 248.5302582, -360.3986816, 346.5692444
4: -60.9292107, 184.6558685, -68.3984451, 207.2056885, -268.1348877, 253.0543213

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4218883, upper bound: 655.4251417
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4218883, upper bound: 655.4251417
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -253.0547791, 507.2885437, -366.4221497, 739.2178345, -992.2725830, 873.7106934
1: -89.3622437, 183.8726501, -130.1624908, 267.8097534, -357.1719971, 314.0350342
2: -55.4284286, 188.6861572, -80.6241760, 274.6762695, -329.8309937, 269.3103333
3: -111.8685608, 220.9114227, -162.1065674, 322.1748657, -433.7084656, 383.0179138
4: -60.9292107, 184.6558685, -88.9131317, 268.1508179, -328.9057922, 273.5690002

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4218883, upper bound: 655.4251417
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4218883, upper bound: 655.4251417
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -249.8941498, 504.0943909, -276.6064148, 553.3734741, -803.2675781, 780.7008057
1: -88.4773712, 182.2526398, -97.4290237, 201.1422272, -289.6195374, 279.6816101
2: -54.6191101, 187.1459045, -60.3096542, 205.9609528, -260.5800781, 247.4555511
3: -110.5838242, 219.0462036, -122.0931168, 241.7169495, -352.3006897, 341.1393127
4: -60.2740059, 183.1118927, -66.4764862, 201.4724884, -261.7464905, 249.5883789

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4220238, upper bound: 655.4257276
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4220238, upper bound: 655.4257276
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -249.8941498, 504.0943909, -358.2796326, 723.8801270, -973.7742920, 862.3740234
1: -88.4773712, 182.2526398, -127.3988800, 262.1683350, -350.6456909, 309.6514587
2: -54.6191101, 187.1459045, -78.8823853, 268.9677734, -323.3601379, 266.0282593
3: -110.5838242, 219.0462036, -158.6021729, 315.4720459, -425.7905579, 377.6483765
4: -60.2740059, 183.1118927, -87.0273056, 262.5148010, -322.6404724, 270.1391907

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4220238, upper bound: 655.4257276
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4220238, upper bound: 655.4257276
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -346.5928040, 701.4409180, -275.0065613, 549.9027100, -896.4954224, 976.4475098
1: -123.5488129, 253.3388672, -96.7794113, 198.4878845, -322.0366821, 350.1182251
2: -76.5876236, 260.2962646, -60.0532379, 203.7509003, -280.3385010, 320.0527344
3: -153.6601868, 304.6977844, -121.1477737, 238.5716400, -392.2318115, 425.5201721
4: -84.3292999, 254.1936493, -66.0137405, 199.7115784, -284.0408936, 319.9804077

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4253752, upper bound: 655.4219384
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4253752, upper bound: 655.4219385
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -346.5928040, 701.4409180, -308.2870789, 618.5671387, -965.1597900, 1009.7279663
1: -123.5488129, 253.3388672, -108.9139404, 223.7356873, -347.2844543, 362.2527161
2: -76.5876236, 260.2962646, -67.4756317, 229.3613586, -305.9489441, 327.4353638
3: -153.6601868, 304.6977844, -136.1895752, 268.7543030, -422.4144592, 440.5665588
4: -84.3292999, 254.1936493, -74.2595062, 224.5109406, -308.8401794, 328.2071228

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4253752, upper bound: 655.4221659
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4253752, upper bound: 655.4221659
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -343.2495117, 697.5584717, -267.7137756, 536.1984253, -879.4479370, 965.2721558
1: -122.4884491, 251.5253296, -94.2826462, 193.4175720, -315.9060059, 345.8079224
2: -75.7326660, 258.5316467, -58.4683723, 198.5819550, -274.3146362, 316.7585449
3: -152.2327576, 302.6506348, -117.9800797, 232.5166168, -384.7493896, 420.3928528
4: -83.6068420, 252.4041290, -64.3059464, 194.6246643, -278.2315063, 316.5561218

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4254888, upper bound: 655.4220163
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4254888, upper bound: 655.4220164
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -343.2495117, 697.5584717, -301.2351990, 605.5251465, -948.7746582, 998.7937012
1: -122.4884491, 251.5253296, -106.5282059, 218.8542938, -341.3427429, 358.0534363
2: -75.7326660, 258.5316467, -65.9451752, 224.3929291, -300.1256104, 324.1954041
3: -152.2327576, 302.6506348, -133.1479797, 262.9236755, -415.1564331, 435.5612488
4: -83.6068420, 252.4041290, -72.6132126, 219.6132202, -303.2200317, 324.8438721

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4254888, upper bound: 655.4222467
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4254888, upper bound: 655.4222467
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -339.9021301, 688.6079712, -294.8830261, 588.4333496, -928.3353882, 983.4909058
1: -121.2832718, 248.8129272, -103.4978409, 213.1597137, -334.4429321, 352.3107605
2: -75.1621323, 255.6735840, -64.1796570, 218.3576202, -293.5197449, 319.6061707
3: -150.7950439, 299.3075562, -129.7318115, 256.2567444, -407.0517273, 428.7431030
4: -82.7853622, 249.5577393, -70.6935501, 213.8545837, -296.6399231, 320.0733948

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4212249, upper bound: 655.4212363
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4212250, upper bound: 655.4212363
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -339.9021301, 688.6079712, -326.8984985, 654.1535645, -994.0556030, 1015.5063477
1: -121.2832718, 248.8129272, -115.2041473, 237.3686829, -358.6518555, 364.0170898
2: -75.1621323, 255.6735840, -71.3340988, 242.9775238, -318.1396179, 326.7227478
3: -150.7950439, 299.3075562, -144.2514954, 285.2517700, -436.0468140, 443.2716980
4: -82.7853622, 249.5577393, -78.6303558, 237.7737122, -320.5590210, 327.9924927

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4212250, upper bound: 655.4212405
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -655.4212249, upper bound: 655.4212405
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -335.9059448, 683.9311523, -286.7716675, 573.1672974, -909.0732422, 970.7028198
1: -120.0460281, 246.6592407, -100.7471695, 207.5525208, -327.5985413, 347.4064026
2: -74.1772232, 253.5373535, -62.4365997, 212.6702576, -286.8474731, 315.7837219
3: -149.1345978, 296.8117371, -126.2434082, 249.5749054, -398.7094727, 422.8480835
4: -81.9281158, 247.4117889, -68.8086853, 208.2351532, -290.1632690, 316.0869141

Time for backsubstitution: 2.30 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.25 + 416.63 = 420.88 seconds
