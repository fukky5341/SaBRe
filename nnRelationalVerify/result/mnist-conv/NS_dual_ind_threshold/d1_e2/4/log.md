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
execution time: IAR + RelationalAnalysis = 23.08 + 33.89 = 56.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.2164975, upper bound: 0.2164976

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2164928, upper bound: 0.2155251
time: 4.08 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2164962, upper bound: 0.2164957
time: 4.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.81 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.81
Output dim: 2, lower bound: -0.2164928, upper bound: 0.2155251
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.81
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

Time for backsubstitution: 20.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 144

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 484

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155251, upper bound: 0.2155251
time: 4.45 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155251, upper bound: 0.2155253
time: 3.83 seconds

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

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 484

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155251, upper bound: 0.2164929
time: 3.87 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155251, upper bound: 0.2164962
time: 4.09 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.70 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.70
Output dim: 2, lower bound: -0.2155251, upper bound: 0.2155251
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.70
Output dim: 2, lower bound: -0.2155251, upper bound: 0.2155253
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.70
Output dim: 2, lower bound: -0.2155251, upper bound: 0.2164929
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.70
Output dim: 2, lower bound: -0.2155251, upper bound: 0.2164962

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

Time for backsubstitution: 21.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 79

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155063, upper bound: 0.2147148
time: 3.78 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155250, upper bound: 0.2155250
time: 4.01 seconds

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

Time for backsubstitution: 21.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 144

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 79

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155063, upper bound: 0.2147148
time: 3.91 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155250, upper bound: 0.2155250
time: 4.04 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -5.5146103, -5.0062675, -5.5138559, -5.0065346, -0.3824987, 0.3633595
1: -4.2234359, -3.6703835, -4.2185984, -3.6710429, -0.3119004, 0.3069882
2: 7.2144537, 7.8699956, 7.2153945, 7.8664403, -0.3827586, 0.3845510
3: -2.3853130, -1.8777966, -2.3839798, -1.8843583, -0.3360648, 0.3389516
4: -12.7614317, -12.0576487, -12.7572937, -12.0582962, -0.3521328, 0.3488271
5: -10.7035112, -10.1200981, -10.6949148, -10.1216898, -0.3324723, 0.3432951
6: -8.0690289, -7.5780759, -8.0648279, -7.5789609, -0.2936473, 0.2892282
7: -8.1379318, -7.5549498, -8.1363173, -7.5559020, -0.3159595, 0.3154905
8: -2.1984129, -1.7333689, -2.1972103, -1.7390623, -0.2568293, 0.2604837
9: -12.3942661, -11.8182430, -12.3915157, -11.8186531, -0.3355274, 0.3327370

Time for backsubstitution: 21.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 144

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 79

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155063, upper bound: 0.2156821
time: 3.86 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155250, upper bound: 0.2164923
time: 3.99 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -5.5146103, -5.0062675, -5.5146103, -5.0062675, -0.3867097, 0.3867097
1: -4.2234359, -3.6703835, -4.2234359, -3.6703835, -0.3094151, 0.3094151
2: 7.2144537, 7.8699956, 7.2144537, 7.8699956, -0.3825932, 0.3825932
3: -2.3853130, -1.8777966, -2.3853130, -1.8777966, -0.3380198, 0.3380198
4: -12.7614317, -12.0576487, -12.7614317, -12.0576487, -0.3503361, 0.3503361
5: -10.7035112, -10.1200981, -10.7035112, -10.1200981, -0.3466868, 0.3466868
6: -8.0690289, -7.5780759, -8.0690289, -7.5780759, -0.2904801, 0.2904801
7: -8.1379318, -7.5549498, -8.1379318, -7.5549498, -0.3160090, 0.3160093
8: -2.1984129, -1.7333689, -2.1984129, -1.7333689, -0.2583811, 0.2583811
9: -12.3942661, -11.8182430, -12.3942661, -11.8182430, -0.3344414, 0.3344414

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 79

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155063, upper bound: 0.2156854
time: 3.60 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155250, upper bound: 0.2164963
time: 3.89 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 35.23 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 35.23
Output dim: 2, lower bound: -0.2155063, upper bound: 0.2147148
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 35.23
Output dim: 2, lower bound: -0.2155250, upper bound: 0.2155250
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 35.23
Output dim: 2, lower bound: -0.2155063, upper bound: 0.2147148
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 35.23
Output dim: 2, lower bound: -0.2155250, upper bound: 0.2155250
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 35.23
Output dim: 2, lower bound: -0.2155063, upper bound: 0.2156821
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 35.23
Output dim: 2, lower bound: -0.2155250, upper bound: 0.2164923
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 35.23
Output dim: 2, lower bound: -0.2155063, upper bound: 0.2156854
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 35.23
Output dim: 2, lower bound: -0.2155250, upper bound: 0.2164963

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

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 79

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2147149
time: 3.79 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2147150
time: 4.02 seconds

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

Time for backsubstitution: 21.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 79

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2155064
time: 3.82 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2155250
time: 3.83 seconds

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

Time for backsubstitution: 21.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 79

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2156824, upper bound: 0.2147147
time: 4.29 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2156824, upper bound: 0.2147147
time: 7.36 seconds

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

Time for backsubstitution: 21.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 79

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2156820, upper bound: 0.2155062
time: 5.39 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2156820, upper bound: 0.2155249
time: 5.49 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.5146098, -5.0062695, -5.5138559, -5.0065346, -0.3824987, 0.3633595
1: -4.2234373, -3.6703858, -4.2185974, -3.6710439, -0.3118994, 0.3069856
2: 7.2144537, 7.8699970, 7.2153945, 7.8664389, -0.3827553, 0.3845491
3: -2.3853135, -1.8777981, -2.3839788, -1.8843586, -0.3360634, 0.3389487
4: -12.7614326, -12.0576458, -12.7572985, -12.0582943, -0.3521326, 0.3488269
5: -10.7035103, -10.1200991, -10.6949148, -10.1216888, -0.3324699, 0.3432937
6: -8.0690231, -7.5780787, -8.0648279, -7.5789604, -0.2936444, 0.2892265
7: -8.1379318, -7.5549507, -8.1363163, -7.5559034, -0.3159592, 0.3154905
8: -2.1984138, -1.7333698, -2.1972084, -1.7390614, -0.2568295, 0.2604830
9: -12.3942671, -11.8182430, -12.3915119, -11.8186512, -0.3355253, 0.3327370

Time for backsubstitution: 21.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 79

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2156821
time: 3.95 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2156821
time: 3.84 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.5146112, -5.0062675, -5.5138559, -5.0065346, -0.3824987, 0.3633590
1: -4.2234378, -3.6703825, -4.2185984, -3.6710429, -0.3119004, 0.3069863
2: 7.2144527, 7.8699956, 7.2153945, 7.8664403, -0.3827548, 0.3845506
3: -2.3853121, -1.8777976, -2.3839798, -1.8843583, -0.3360653, 0.3389502
4: -12.7614317, -12.0576477, -12.7572937, -12.0582962, -0.3521328, 0.3488259
5: -10.7035122, -10.1200972, -10.6949148, -10.1216898, -0.3324718, 0.3432970
6: -8.0690269, -7.5780764, -8.0648279, -7.5789609, -0.2936449, 0.2892280
7: -8.1379318, -7.5549517, -8.1363173, -7.5559020, -0.3159595, 0.3154905
8: -2.1984158, -1.7333689, -2.1972103, -1.7390623, -0.2568295, 0.2604833
9: -12.3942661, -11.8182459, -12.3915157, -11.8186531, -0.3355262, 0.3327367

Time for backsubstitution: 21.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 79

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2164735
time: 4.00 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2164923
time: 4.00 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.5146098, -5.0062695, -5.5146103, -5.0062675, -0.3867092, 0.3867102
1: -4.2234373, -3.6703858, -4.2234354, -3.6703858, -0.3094149, 0.3094125
2: 7.2144537, 7.8699970, 7.2144547, 7.8699951, -0.3825903, 0.3825932
3: -2.3853135, -1.8777981, -2.3853137, -1.8777980, -0.3380194, 0.3380184
4: -12.7614326, -12.0576458, -12.7614317, -12.0576496, -0.3503356, 0.3503358
5: -10.7035103, -10.1200991, -10.7035112, -10.1201019, -0.3466907, 0.3466859
6: -8.0690231, -7.5780787, -8.0690269, -7.5780754, -0.2904775, 0.2904787
7: -8.1379318, -7.5549507, -8.1379318, -7.5549521, -0.3160086, 0.3160098
8: -2.1984138, -1.7333698, -2.1984129, -1.7333698, -0.2583804, 0.2583807
9: -12.3942671, -11.8182430, -12.3942680, -11.8182430, -0.3344398, 0.3344407

Time for backsubstitution: 21.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 79

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147227, upper bound: 0.2156852
time: 4.63 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147227, upper bound: 0.2156857
time: 4.43 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.5146112, -5.0062675, -5.5146103, -5.0062675, -0.3867106, 0.3867092
1: -4.2234378, -3.6703825, -4.2234359, -3.6703835, -0.3094149, 0.3094134
2: 7.2144527, 7.8699956, 7.2144537, 7.8699956, -0.3825898, 0.3825936
3: -2.3853121, -1.8777976, -2.3853130, -1.8777966, -0.3380198, 0.3380189
4: -12.7614317, -12.0576477, -12.7614317, -12.0576487, -0.3503361, 0.3503344
5: -10.7035122, -10.1200972, -10.7035112, -10.1200981, -0.3466873, 0.3466897
6: -8.0690269, -7.5780764, -8.0690289, -7.5780759, -0.2904778, 0.2904801
7: -8.1379318, -7.5549517, -8.1379318, -7.5549498, -0.3160090, 0.3160093
8: -2.1984158, -1.7333689, -2.1984129, -1.7333689, -0.2583814, 0.2583811
9: -12.3942661, -11.8182459, -12.3942661, -11.8182430, -0.3344405, 0.3344414

Time for backsubstitution: 21.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 144

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 79

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147227, upper bound: 0.2164773
time: 3.88 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147227, upper bound: 0.2164963
time: 3.95 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.63 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2147149
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2147150
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2155064
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2155250
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2156824, upper bound: 0.2147147
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2156824, upper bound: 0.2147147
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2156820, upper bound: 0.2155062
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2156820, upper bound: 0.2155249
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2156821
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2156821
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2164735
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2147148, upper bound: 0.2164923
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2147227, upper bound: 0.2156852
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2147227, upper bound: 0.2156857
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2147227, upper bound: 0.2164773
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.63
Output dim: 2, lower bound: -0.2147227, upper bound: 0.2164963

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

Time for backsubstitution: 21.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 79

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 79

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

Time for backsubstitution: 21.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 79

Time for candidate selection: 0.11 seconds

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

Time for backsubstitution: 21.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 79

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 79

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

Time for backsubstitution: 21.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 79

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 79

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

Time for backsubstitution: 21.97 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.97 + 563.10 = 620.07 seconds
