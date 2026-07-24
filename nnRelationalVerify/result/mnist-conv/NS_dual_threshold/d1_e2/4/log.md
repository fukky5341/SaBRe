## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.214332525


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3835130, 0.3835125)
1: (-4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3126643, 0.3126643)
2: (7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3854012, 0.3854012)
3: (-2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3425937, 0.3425937)
4: (-12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3527765, 0.3527765)
5: (-10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3518991, 0.3518991)
6: (-8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2934105, 0.2934110)
7: (-8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3171108, 0.3171108)
8: (-2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2629416, 0.2629416)
9: (-12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3359382, 0.3359382)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.53 + 33.54 = 56.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.2164975, upper bound: 0.2164976

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 484
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2164928, upper bound: 0.2155251
time: 3.98 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2164962, upper bound: 0.2164957
time: 4.41 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.54 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.54
Output dim: 2, lower bound: -0.2164928, upper bound: 0.2155251
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.54
Output dim: 2, lower bound: -0.2164962, upper bound: 0.2164957

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -5.5138559, -5.0065346, -5.5144138, -5.0063252, -0.3633051, 0.3827138
1: -4.2185984, -3.6710429, -4.2223015, -3.6703882, -0.3069854, 0.3105953
2: 7.2153945, 7.8664403, 7.2145424, 7.8691707, -0.3836851, 0.3826733
3: -2.3839798, -1.8843583, -2.3850684, -1.8793150, -0.3372436, 0.3359141
4: -12.7572937, -12.0582962, -12.7604799, -12.0576801, -0.3487954, 0.3511858
5: -10.6949148, -10.1216898, -10.7015114, -10.1202946, -0.3432298, 0.3313727
6: -8.0648279, -7.5789609, -8.0680542, -7.5781393, -0.2891228, 0.2926154
7: -8.1363173, -7.5559020, -8.1375637, -7.5551171, -0.3153598, 0.3153651
8: -2.1972103, -1.7390623, -2.1983433, -1.7347212, -0.2603691, 0.2567756
9: -12.3915157, -11.8186531, -12.3935986, -11.8182678, -0.3327160, 0.3346937

Time for backsubstitution: 20.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 484

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155251, upper bound: 0.2155251
time: 4.23 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155251, upper bound: 0.2155253
time: 3.62 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -5.5146103, -5.0062675, -5.5146112, -5.0062685, -0.3867097, 0.3831410
1: -4.2234359, -3.6703835, -4.2234378, -3.6703835, -0.3094153, 0.3126194
2: 7.2144537, 7.8699956, 7.2144527, 7.8699956, -0.3854017, 0.3825934
3: -2.3853130, -1.8777966, -2.3853123, -1.8777944, -0.3425932, 0.3380208
4: -12.7614317, -12.0576487, -12.7614355, -12.0576487, -0.3503361, 0.3527763
5: -10.7035112, -10.1200981, -10.7035141, -10.1200981, -0.3466868, 0.3518991
6: -8.0690289, -7.5780759, -8.0690289, -7.5780745, -0.2904801, 0.2934113
7: -8.1379318, -7.5549498, -8.1379309, -7.5549507, -0.3160098, 0.3171105
8: -2.1984129, -1.7333689, -2.1984138, -1.7333698, -0.2629414, 0.2583814
9: -12.3942661, -11.8182430, -12.3942699, -11.8182421, -0.3344414, 0.3359320

Time for backsubstitution: 21.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 79

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2164923, upper bound: 0.2154816
time: 4.42 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2162283, upper bound: 0.2162278
time: 3.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.43 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.43
Output dim: 2, lower bound: -0.2155251, upper bound: 0.2155251
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.43
Output dim: 2, lower bound: -0.2155251, upper bound: 0.2155253
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 29.43
Output dim: 2, lower bound: -0.2164923, upper bound: 0.2154816
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 29.43
Output dim: 2, lower bound: -0.2162283, upper bound: 0.2162278

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -5.5138559, -5.0065346, -5.5138559, -5.0065346, -0.3631206, 0.3631206
1: -4.2185984, -3.6710429, -4.2185984, -3.6710429, -0.3062673, 0.3062673
2: 7.2153945, 7.8664403, 7.2153945, 7.8664403, -0.3819141, 0.3819141
3: -2.3839798, -1.8843583, -2.3839798, -1.8843583, -0.3324404, 0.3324404
4: -12.7572937, -12.0582962, -12.7572937, -12.0582962, -0.3481851, 0.3481851
5: -10.6949148, -10.1216898, -10.6949148, -10.1216898, -0.3247728, 0.3247728
6: -8.0648279, -7.5789609, -8.0648279, -7.5789609, -0.2895036, 0.2895033
7: -8.1363173, -7.5559020, -8.1363173, -7.5559020, -0.3143458, 0.3143458
8: -2.1972103, -1.7390623, -2.1972103, -1.7390623, -0.2556419, 0.2556419
9: -12.3915157, -11.8186531, -12.3915157, -11.8186531, -0.3323462, 0.3323462

Time for backsubstitution: 21.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 79

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 79

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155063, upper bound: 0.2147148
time: 3.85 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155250, upper bound: 0.2155250
time: 3.85 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -5.5138559, -5.0065346, -5.5146103, -5.0062675, -0.3633595, 0.3824987
1: -4.2185984, -3.6710429, -4.2234359, -3.6703835, -0.3069882, 0.3119004
2: 7.2153945, 7.8664403, 7.2144537, 7.8699956, -0.3845510, 0.3827586
3: -2.3839798, -1.8843583, -2.3853130, -1.8777966, -0.3389516, 0.3360648
4: -12.7572937, -12.0582962, -12.7614317, -12.0576487, -0.3488271, 0.3521328
5: -10.6949148, -10.1216898, -10.7035112, -10.1200981, -0.3432951, 0.3324723
6: -8.0648279, -7.5789609, -8.0690289, -7.5780759, -0.2892282, 0.2936473
7: -8.1363173, -7.5559020, -8.1379318, -7.5549498, -0.3154900, 0.3159595
8: -2.1972103, -1.7390623, -2.1984129, -1.7333689, -0.2604836, 0.2568293
9: -12.3915157, -11.8186531, -12.3942661, -11.8182430, -0.3327370, 0.3355274

Time for backsubstitution: 21.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 79

## Relational analysis of NS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 79

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155063, upper bound: 0.2147148
time: 3.80 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155250, upper bound: 0.2155250
time: 4.19 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -5.5146103, -5.0063295, -5.5146112, -5.0062685, -0.3867097, 0.3831387
1: -4.2234364, -3.6703835, -4.2234378, -3.6703835, -0.3094158, 0.3126175
2: 7.2144527, 7.8699341, 7.2144527, 7.8699956, -0.3854008, 0.3825915
3: -2.3853135, -1.8778552, -2.3853123, -1.8777944, -0.3425932, 0.3379631
4: -12.7614307, -12.0576506, -12.7614355, -12.0576487, -0.3503361, 0.3527746
5: -10.7031622, -10.1200972, -10.7035141, -10.1200981, -0.3462968, 0.3518982
6: -8.0687084, -7.5780754, -8.0690289, -7.5780745, -0.2901564, 0.2934113
7: -8.1379318, -7.5549569, -8.1379309, -7.5549507, -0.3160102, 0.3171062
8: -2.1984138, -1.7333679, -2.1984138, -1.7333698, -0.2629411, 0.2583809
9: -12.3942680, -11.8182459, -12.3942699, -11.8182421, -0.3344405, 0.3359284

Time for backsubstitution: 21.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 79

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154819, upper bound: 0.2154818
time: 4.06 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154819, upper bound: 0.2154818
time: 3.86 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -5.5148873, -5.0062704, -5.5145998, -5.0062699, -0.3869872, 0.3833623
1: -4.2233677, -3.6706810, -4.2234340, -3.6704712, -0.3097181, 0.3127136
2: 7.2141657, 7.8699899, 7.2144642, 7.8699942, -0.3856688, 0.3829215
3: -2.3855581, -1.8778009, -2.3853116, -1.8777976, -0.3428473, 0.3380370
4: -12.7613354, -12.0581074, -12.7614307, -12.0577822, -0.3501136, 0.3528175
5: -10.7035065, -10.1185389, -10.7035103, -10.1201077, -0.3467407, 0.3534541
6: -8.0689831, -7.5766168, -8.0690165, -7.5780764, -0.2905402, 0.2948728
7: -8.1377764, -7.5557542, -8.1379309, -7.5551853, -0.3166840, 0.3173020
8: -2.1984138, -1.7334013, -2.1984158, -1.7333727, -0.2630720, 0.2583523
9: -12.3941259, -11.8188076, -12.3942661, -11.8184090, -0.3349988, 0.3362551

Time for backsubstitution: 21.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 484

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 79

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154819, upper bound: 0.2162278
time: 4.41 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154819, upper bound: 0.2162280
time: 3.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.63 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 2, lower bound: -0.2155063, upper bound: 0.2147148
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 2, lower bound: -0.2155250, upper bound: 0.2155250
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 2, lower bound: -0.2155063, upper bound: 0.2147148
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 2, lower bound: -0.2155250, upper bound: 0.2155250
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 2, lower bound: -0.2154819, upper bound: 0.2154818
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 2, lower bound: -0.2154819, upper bound: 0.2154818
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 2, lower bound: -0.2154819, upper bound: 0.2162278
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 2, lower bound: -0.2154819, upper bound: 0.2162280

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.5138550, -5.0065346, -5.5138559, -5.0065346, -0.3631201, 0.3631206
1: -4.2185960, -3.6710443, -4.2185974, -3.6710439, -0.3062665, 0.3062651
2: 7.2153969, 7.8664422, 7.2153945, 7.8664389, -0.3819098, 0.3819122
3: -2.3839798, -1.8843600, -2.3839788, -1.8843586, -0.3324389, 0.3324380
4: -12.7572994, -12.0582952, -12.7572985, -12.0582943, -0.3481841, 0.3481846
5: -10.6949120, -10.1216908, -10.6949148, -10.1216888, -0.3247762, 0.3247728
6: -8.0648270, -7.5789595, -8.0648279, -7.5789604, -0.2895012, 0.2895024
7: -8.1363173, -7.5559020, -8.1363163, -7.5559034, -0.3143456, 0.3143454
8: -2.1972065, -1.7390604, -2.1972084, -1.7390614, -0.2556424, 0.2556415
9: -12.3915110, -11.8186531, -12.3915119, -11.8186512, -0.3323448, 0.3323457

Time for backsubstitution: 21.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 79

### Candidate
type: A, layer: 1, pos: 79

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2147149
time: 3.71 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2147150
time: 3.89 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.5138559, -5.0065346, -5.5138559, -5.0065346, -0.3631201, 0.3631206
1: -4.2185974, -3.6710439, -4.2185984, -3.6710429, -0.3062670, 0.3062658
2: 7.2153940, 7.8664408, 7.2153945, 7.8664403, -0.3819098, 0.3819146
3: -2.3839788, -1.8843583, -2.3839798, -1.8843583, -0.3324394, 0.3324389
4: -12.7572956, -12.0582972, -12.7572937, -12.0582962, -0.3481853, 0.3481843
5: -10.6949148, -10.1216898, -10.6949148, -10.1216898, -0.3247728, 0.3247747
6: -8.0648308, -7.5789604, -8.0648279, -7.5789609, -0.2895012, 0.2895036
7: -8.1363173, -7.5559030, -8.1363173, -7.5559020, -0.3143458, 0.3143451
8: -2.1972094, -1.7390614, -2.1972103, -1.7390623, -0.2556422, 0.2556417
9: -12.3915138, -11.8186512, -12.3915157, -11.8186531, -0.3323455, 0.3323457

Time for backsubstitution: 21.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 144

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 79

### Candidate
type: A, layer: 1, pos: 79

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2155064
time: 3.79 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2155250
time: 3.88 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.5138550, -5.0065346, -5.5146103, -5.0062675, -0.3633595, 0.3824997
1: -4.2185960, -3.6710443, -4.2234354, -3.6703858, -0.3069870, 0.3118987
2: 7.2153969, 7.8664422, 7.2144547, 7.8699951, -0.3845472, 0.3827577
3: -2.3839798, -1.8843600, -2.3853137, -1.8777980, -0.3389506, 0.3360620
4: -12.7572994, -12.0582952, -12.7614317, -12.0576496, -0.3488266, 0.3521328
5: -10.6949120, -10.1216908, -10.7035112, -10.1201019, -0.3432989, 0.3324690
6: -8.0648270, -7.5789595, -8.0690269, -7.5780754, -0.2892249, 0.2936461
7: -8.1363173, -7.5559020, -8.1379318, -7.5549521, -0.3154898, 0.3159595
8: -2.1972065, -1.7390604, -2.1984129, -1.7333698, -0.2604829, 0.2568290
9: -12.3915110, -11.8186531, -12.3942680, -11.8182430, -0.3327358, 0.3355260

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 79

### Candidate
type: A, layer: 1, pos: 79

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2156824, upper bound: 0.2147147
time: 4.34 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2156824, upper bound: 0.2147147
time: 7.30 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.5138559, -5.0065346, -5.5146103, -5.0062675, -0.3633590, 0.3824978
1: -4.2185974, -3.6710439, -4.2234359, -3.6703835, -0.3069878, 0.3118989
2: 7.2153940, 7.8664408, 7.2144537, 7.8699956, -0.3845468, 0.3827586
3: -2.3839788, -1.8843583, -2.3853130, -1.8777966, -0.3389506, 0.3360634
4: -12.7572956, -12.0582972, -12.7614317, -12.0576487, -0.3488274, 0.3521311
5: -10.6949148, -10.1216898, -10.7035112, -10.1200981, -0.3432946, 0.3324702
6: -8.0648308, -7.5789604, -8.0690289, -7.5780759, -0.2892261, 0.2936473
7: -8.1363173, -7.5559030, -8.1379318, -7.5549498, -0.3154905, 0.3159585
8: -2.1972094, -1.7390614, -2.1984129, -1.7333689, -0.2604837, 0.2568290
9: -12.3915138, -11.8186512, -12.3942661, -11.8182430, -0.3327365, 0.3355272

Time for backsubstitution: 21.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 79

### Candidate
type: A, layer: 1, pos: 79

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2156820, upper bound: 0.2155062
time: 5.16 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2156820, upper bound: 0.2155249
time: 5.50 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -5.5146103, -5.0063295, -5.5146108, -5.0063305, -0.3866153, 0.3831372
1: -4.2234364, -3.6703835, -4.2234373, -3.6703844, -0.3093462, 0.3126175
2: 7.2144527, 7.8699341, 7.2144527, 7.8699336, -0.3852854, 0.3825905
3: -2.3853135, -1.8778552, -2.3853135, -1.8778534, -0.3425341, 0.3379626
4: -12.7614307, -12.0576506, -12.7614346, -12.0576496, -0.3503332, 0.3527746
5: -10.7031622, -10.1200972, -10.7031641, -10.1200981, -0.3462963, 0.3515077
6: -8.0687084, -7.5780754, -8.0687084, -7.5780764, -0.2901556, 0.2930863
7: -8.1379318, -7.5549569, -8.1379328, -7.5549564, -0.3160052, 0.3171065
8: -2.1984138, -1.7333679, -2.1984129, -1.7333698, -0.2629242, 0.2583799
9: -12.3942680, -11.8182459, -12.3942699, -11.8182459, -0.3342853, 0.3359077

Time for backsubstitution: 21.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 484

## Relational analysis of NS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2156221, upper bound: 0.2154628
time: 3.44 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2164324, upper bound: 0.2154817
time: 3.63 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -5.5146103, -5.0063295, -5.5148859, -5.0062714, -0.3866773, 0.3834152
1: -4.2234364, -3.6703835, -4.2233696, -3.6706820, -0.3094137, 0.3125558
2: 7.2144527, 7.8699341, 7.2141638, 7.8699932, -0.3853397, 0.3828623
3: -2.3853135, -1.8778552, -2.3855574, -1.8777995, -0.3425922, 0.3382168
4: -12.7614307, -12.0576506, -12.7613344, -12.0581083, -0.3498781, 0.3526843
5: -10.7031622, -10.1200972, -10.7035084, -10.1185398, -0.3478527, 0.3518548
6: -8.0687084, -7.5780754, -8.0689831, -7.5766182, -0.2916274, 0.2933707
7: -8.1379318, -7.5549569, -8.1377735, -7.5557528, -0.3158450, 0.3169546
8: -2.1984138, -1.7333679, -2.1984138, -1.7333975, -0.2628875, 0.2583785
9: -12.3942680, -11.8182459, -12.3941278, -11.8188086, -0.3343985, 0.3357592

Time for backsubstitution: 21.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 484

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2156221, upper bound: 0.2154628
time: 3.87 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2164324, upper bound: 0.2154817
time: 3.60 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -5.5148873, -5.0062704, -5.5146108, -5.0063305, -0.3868928, 0.3831043
1: -4.2233677, -3.6706810, -4.2234373, -3.6703844, -0.3096490, 0.3123236
2: 7.2141657, 7.8699899, 7.2144527, 7.8699336, -0.3855567, 0.3824637
3: -2.3855581, -1.8778009, -2.3853135, -1.8778534, -0.3427887, 0.3380203
4: -12.7613354, -12.0581074, -12.7614346, -12.0576496, -0.3502438, 0.3523192
5: -10.7035065, -10.1185389, -10.7031641, -10.1200981, -0.3466425, 0.3530650
6: -8.0689831, -7.5766168, -8.0687084, -7.5780764, -0.2904394, 0.2945580
7: -8.1377764, -7.5557542, -8.1379328, -7.5549564, -0.3168311, 0.3163064
8: -2.1984138, -1.7334013, -2.1984129, -1.7333698, -0.2629230, 0.2583833
9: -12.3941259, -11.8188076, -12.3942699, -11.8182459, -0.3348491, 0.3353465

Time for backsubstitution: 21.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 484

## Relational analysis of NS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2146715, upper bound: 0.2162082
time: 4.45 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154818, upper bound: 0.2162273
time: 6.25 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -5.5148873, -5.0062704, -5.5148859, -5.0062714, -0.3863230, 0.3832626
1: -4.2233677, -3.6706810, -4.2233696, -3.6706820, -0.3092473, 0.3127108
2: 7.2141657, 7.8699899, 7.2141638, 7.8699932, -0.3851581, 0.3827548
3: -2.3855581, -1.8778009, -2.3855574, -1.8777995, -0.3426204, 0.3380485
4: -12.7613354, -12.0581074, -12.7613344, -12.0581083, -0.3501313, 0.3528039
5: -10.7035065, -10.1185389, -10.7035084, -10.1185398, -0.3466206, 0.3518305
6: -8.0689831, -7.5766168, -8.0689831, -7.5766182, -0.2906961, 0.2936289
7: -8.1377764, -7.5557542, -8.1377735, -7.5557528, -0.3157797, 0.3172929
8: -2.1984138, -1.7334013, -2.1984138, -1.7333975, -0.2630494, 0.2583494
9: -12.3941259, -11.8188076, -12.3941278, -11.8188086, -0.3340406, 0.3362336

Time for backsubstitution: 21.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 484

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2146715, upper bound: 0.2162087
time: 3.92 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154818, upper bound: 0.2162279
time: 4.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.68 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2147149
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2147150
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2155064
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2155250
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2156824, upper bound: 0.2147147
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2156824, upper bound: 0.2147147
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2156820, upper bound: 0.2155062
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2156820, upper bound: 0.2155249
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2156221, upper bound: 0.2154628
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2164324, upper bound: 0.2154817
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2156221, upper bound: 0.2154628
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2164324, upper bound: 0.2154817
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2146715, upper bound: 0.2162082
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2154818, upper bound: 0.2162273
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2146715, upper bound: 0.2162087
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.68
Output dim: 2, lower bound: -0.2154818, upper bound: 0.2162279

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.5138550, -5.0065346, -5.5138550, -5.0065346, -0.3631206, 0.3631208
1: -4.2185960, -3.6710443, -4.2185960, -3.6710443, -0.3062649, 0.3062649
2: 7.2153969, 7.8664422, 7.2153969, 7.8664422, -0.3819098, 0.3819098
3: -2.3839798, -1.8843600, -2.3839798, -1.8843600, -0.3324375, 0.3324375
4: -12.7572994, -12.0582952, -12.7572994, -12.0582952, -0.3481843, 0.3481843
5: -10.6949120, -10.1216908, -10.6949120, -10.1216908, -0.3247762, 0.3247762
6: -8.0648270, -7.5789595, -8.0648270, -7.5789595, -0.2895007, 0.2895005
7: -8.1363173, -7.5559020, -8.1363173, -7.5559020, -0.3143449, 0.3143451
8: -2.1972065, -1.7390604, -2.1972065, -1.7390604, -0.2556422, 0.2556422
9: -12.3915110, -11.8186531, -12.3915110, -11.8186531, -0.3323448, 0.3323448

Time for backsubstitution: 21.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 79

### Candidate
type: B, layer: 1, pos: 79

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.5138550, -5.0065346, -5.5138559, -5.0065346, -0.3631210, 0.3631201
1: -4.2185960, -3.6710443, -4.2185974, -3.6710439, -0.3062670, 0.3062649
2: 7.2153969, 7.8664422, 7.2153940, 7.8664408, -0.3819103, 0.3819127
3: -2.3839798, -1.8843600, -2.3839788, -1.8843583, -0.3324404, 0.3324370
4: -12.7572994, -12.0582952, -12.7572956, -12.0582972, -0.3481846, 0.3481853
5: -10.6949120, -10.1216908, -10.6949148, -10.1216898, -0.3247752, 0.3247728
6: -8.0648270, -7.5789595, -8.0648308, -7.5789604, -0.2895010, 0.2895031
7: -8.1363173, -7.5559020, -8.1363173, -7.5559030, -0.3143449, 0.3143461
8: -2.1972065, -1.7390604, -2.1972094, -1.7390614, -0.2556422, 0.2556415
9: -12.3915110, -11.8186531, -12.3915138, -11.8186512, -0.3323445, 0.3323460

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 79

### Candidate
type: A, layer: 1, pos: 79

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.5138559, -5.0065346, -5.5138550, -5.0065346, -0.3631201, 0.3631208
1: -4.2185974, -3.6710439, -4.2185960, -3.6710443, -0.3062649, 0.3062670
2: 7.2153940, 7.8664408, 7.2153969, 7.8664422, -0.3819127, 0.3819103
3: -2.3839788, -1.8843583, -2.3839798, -1.8843600, -0.3324370, 0.3324409
4: -12.7572956, -12.0582972, -12.7572994, -12.0582952, -0.3481853, 0.3481846
5: -10.6949148, -10.1216898, -10.6949120, -10.1216908, -0.3247728, 0.3247752
6: -8.0648308, -7.5789604, -8.0648270, -7.5789595, -0.2895031, 0.2895010
7: -8.1363173, -7.5559030, -8.1363173, -7.5559020, -0.3143458, 0.3143449
8: -2.1972094, -1.7390614, -2.1972065, -1.7390604, -0.2556415, 0.2556422
9: -12.3915138, -11.8186512, -12.3915110, -11.8186531, -0.3323460, 0.3323445

Time for backsubstitution: 21.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 79

### Candidate
type: B, layer: 1, pos: 79

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.5138559, -5.0065346, -5.5138559, -5.0065346, -0.3631201, 0.3631198
1: -4.2185974, -3.6710439, -4.2185974, -3.6710439, -0.3062654, 0.3062654
2: 7.2153940, 7.8664408, 7.2153940, 7.8664408, -0.3819098, 0.3819101
3: -2.3839788, -1.8843583, -2.3839788, -1.8843583, -0.3324389, 0.3324394
4: -12.7572956, -12.0582972, -12.7572956, -12.0582972, -0.3481836, 0.3481836
5: -10.6949148, -10.1216898, -10.6949148, -10.1216898, -0.3247747, 0.3247747
6: -8.0648308, -7.5789604, -8.0648308, -7.5789604, -0.2895014, 0.2895014
7: -8.1363173, -7.5559030, -8.1363173, -7.5559030, -0.3143451, 0.3143451
8: -2.1972094, -1.7390614, -2.1972094, -1.7390614, -0.2556424, 0.2556424
9: -12.3915138, -11.8186512, -12.3915138, -11.8186512, -0.3323457, 0.3323457

Time for backsubstitution: 20.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 79

### Candidate
type: B, layer: 1, pos: 79

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.5138550, -5.0065346, -5.5146098, -5.0062695, -0.3633595, 0.3824997
1: -4.2185960, -3.6710443, -4.2234373, -3.6703858, -0.3069854, 0.3118980
2: 7.2153969, 7.8664422, 7.2144537, 7.8699970, -0.3845468, 0.3827546
3: -2.3839798, -1.8843600, -2.3853135, -1.8777981, -0.3389482, 0.3360615
4: -12.7572994, -12.0582952, -12.7614326, -12.0576458, -0.3488266, 0.3521326
5: -10.6949120, -10.1216908, -10.7035103, -10.1200991, -0.3432984, 0.3324685
6: -8.0648270, -7.5789595, -8.0690231, -7.5780787, -0.2892246, 0.2936437
7: -8.1363173, -7.5559020, -8.1379318, -7.5549507, -0.3154902, 0.3159590
8: -2.1972065, -1.7390604, -2.1984138, -1.7333698, -0.2604825, 0.2568293
9: -12.3915110, -11.8186531, -12.3942671, -11.8182430, -0.3327360, 0.3355253

Time for backsubstitution: 21.55 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.07 + 556.93 = 612.99 seconds
