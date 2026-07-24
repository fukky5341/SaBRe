## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.091838112


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.3866034, -9.6609735, -10.3866034, -9.6609735, -0.5056384, 0.5056384)
1: (3.7886105, 4.2537642, 3.7886105, 4.2537642, -0.2166868, 0.2166868)
2: (-4.0563807, -3.5707655, -4.0563807, -3.5707655, -0.2312573, 0.2312573)
3: (-12.0196276, -11.4445496, -12.0196276, -11.4445496, -0.2746441, 0.2746441)
4: (-2.2219760, -1.7096756, -2.2219760, -1.7096756, -0.2315657, 0.2315657)
5: (-9.8138046, -9.2945690, -9.8138046, -9.2945690, -0.1551014, 0.1551014)
6: (-6.6133256, -5.9296665, -6.6133256, -5.9296665, -0.3535559, 0.3535559)
7: (-3.1167557, -2.6428428, -3.1167557, -2.6428428, -0.2603815, 0.2603815)
8: (-2.9814692, -2.5360689, -2.9814692, -2.5360689, -0.2330493, 0.2330492)
9: (-12.8839960, -12.3250265, -12.8839960, -12.3250265, -0.3167467, 0.3167470)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.00 + 32.93 = 54.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0956647, upper bound: 0.0956647

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 456
type: A, layer: 1, pos: 51

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 456

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0956364, upper bound: 0.0947919
time: 3.57 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0956637, upper bound: 0.0956634
time: 3.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.97 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.97
Output dim: 1, lower bound: -0.0956364, upper bound: 0.0947919
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.97
Output dim: 1, lower bound: -0.0956637, upper bound: 0.0956634

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -10.3805342, -9.6672621, -10.3861217, -9.6647015, -0.4954925, 0.4985721
1: 3.7928257, 4.2506814, 3.7892747, 4.2518988, -0.2094377, 0.2123996
2: -4.0519228, -3.5752499, -4.0556154, -3.5734248, -0.2249845, 0.2265024
3: -12.0167112, -11.4471836, -12.0178509, -11.4451904, -0.2713056, 0.2694397
4: -2.2193990, -1.7117391, -2.2205768, -1.7099718, -0.2291170, 0.2277164
5: -9.8093872, -9.2999983, -9.8133125, -9.2978048, -0.1479954, 0.1495734
6: -6.6099677, -5.9316015, -6.6127281, -5.9307866, -0.3489254, 0.3508275
7: -3.1141734, -2.6453691, -3.1151876, -2.6435390, -0.2548048, 0.2536478
8: -2.9752908, -2.5409517, -2.9778914, -2.5364342, -0.2265167, 0.2246323
9: -12.8800182, -12.3303165, -12.8835287, -12.3278675, -0.3096752, 0.3115160

Time for backsubstitution: 20.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 456
type: B, layer: 1, pos: 51

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 456

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0947922, upper bound: 0.0947920
time: 4.06 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0947922, upper bound: 0.0947920
time: 3.34 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -10.3866034, -9.6609774, -10.3866024, -9.6609774, -0.5056355, 0.5005898
1: 3.7886119, 4.2537613, 3.7886119, 4.2537622, -0.2154517, 0.2145901
2: -4.0563812, -3.5707695, -4.0563817, -3.5707676, -0.2312560, 0.2288606
3: -12.0196257, -11.4445486, -12.0196266, -11.4445486, -0.2750311, 0.2732072
4: -2.2219746, -1.7096759, -2.2219760, -1.7096757, -0.2312925, 0.2323762
5: -9.8138056, -9.2945709, -9.8138046, -9.2945709, -0.1529148, 0.1485732
6: -6.6133256, -5.9296675, -6.6133256, -5.9296675, -0.3528767, 0.3522468
7: -3.1167550, -2.6428435, -3.1167557, -2.6428428, -0.2625122, 0.2546124
8: -2.9814658, -2.5360699, -2.9814672, -2.5360680, -0.2275565, 0.2330472
9: -12.8839951, -12.3250294, -12.8839970, -12.3250256, -0.3172944, 0.3165603

Time for backsubstitution: 21.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 456
type: B, layer: 1, pos: 51

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 456

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0947921, upper bound: 0.0956364
time: 3.07 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0947921, upper bound: 0.0956364
time: 3.18 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.79 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.79
Output dim: 1, lower bound: -0.0947922, upper bound: 0.0947920
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.79
Output dim: 1, lower bound: -0.0947922, upper bound: 0.0947920
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.79
Output dim: 1, lower bound: -0.0947921, upper bound: 0.0956364
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.79
Output dim: 1, lower bound: -0.0947921, upper bound: 0.0956364

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -10.3805342, -9.6672621, -10.3805342, -9.6672621, -0.4929528, 0.4929523
1: 3.7928257, 4.2506814, 3.7928257, 4.2506814, -0.2079912, 0.2079914
2: -4.0519228, -3.5752499, -4.0519228, -3.5752499, -0.2231562, 0.2231562
3: -12.0167112, -11.4471836, -12.0167112, -11.4471836, -0.2681947, 0.2681947
4: -2.2193990, -1.7117391, -2.2193990, -1.7117391, -0.2270428, 0.2270428
5: -9.8093872, -9.2999983, -9.8093872, -9.2999983, -0.1458419, 0.1458420
6: -6.6099677, -5.9316015, -6.6099677, -5.9316015, -0.3480980, 0.3480978
7: -3.1141734, -2.6453691, -3.1141734, -2.6453691, -0.2496932, 0.2496934
8: -2.9752908, -2.5409517, -2.9752908, -2.5409517, -0.2220256, 0.2220257
9: -12.8800182, -12.3303165, -12.8800182, -12.3303165, -0.3079581, 0.3079581

Time for backsubstitution: 21.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 51

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 51

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -10.3805342, -9.6672621, -10.3863249, -9.6609774, -0.4992292, 0.4989619
1: 3.7928257, 4.2506814, 3.7888761, 4.2537613, -0.2109950, 0.2120388
2: -4.0519228, -3.5752499, -4.0560131, -3.5707715, -0.2276781, 0.2265670
3: -12.0167112, -11.4471836, -12.0196257, -11.4449110, -0.2708654, 0.2700336
4: -2.2193990, -1.7117391, -2.2219753, -1.7098910, -0.2285566, 0.2294842
5: -9.8093872, -9.2999983, -9.8138056, -9.2946243, -0.1483728, 0.1475250
6: -6.6099677, -5.9316015, -6.6130724, -5.9296679, -0.3500628, 0.3507047
7: -3.1141734, -2.6453691, -3.1167550, -2.6430869, -0.2522209, 0.2513032
8: -2.9752908, -2.5409517, -2.9814634, -2.5361881, -0.2267470, 0.2280757
9: -12.8800182, -12.3303165, -12.8839493, -12.3250294, -0.3129075, 0.3114357

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 51

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 51

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -10.3863249, -9.6609774, -10.3805342, -9.6672621, -0.4989619, 0.4992290
1: 3.7888761, 4.2537613, 3.7928257, 4.2506814, -0.2120388, 0.2109950
2: -4.0560131, -3.5707715, -4.0519228, -3.5752499, -0.2265670, 0.2276781
3: -12.0196257, -11.4449110, -12.0167112, -11.4471836, -0.2700334, 0.2708652
4: -2.2219753, -1.7098910, -2.2193990, -1.7117391, -0.2294842, 0.2285568
5: -9.8138056, -9.2946243, -9.8093872, -9.2999983, -0.1475250, 0.1483728
6: -6.6130724, -5.9296679, -6.6099677, -5.9316015, -0.3507047, 0.3500628
7: -3.1167550, -2.6430869, -3.1141734, -2.6453691, -0.2513027, 0.2522209
8: -2.9814634, -2.5361881, -2.9752908, -2.5409517, -0.2280757, 0.2267470
9: -12.8839493, -12.3250294, -12.8800182, -12.3303165, -0.3114357, 0.3129075

Time for backsubstitution: 22.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 51

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 51

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -10.3866034, -9.6609774, -10.3866034, -9.6609774, -0.5005891, 0.5005889
1: 3.7886119, 4.2537613, 3.7886119, 4.2537613, -0.2145896, 0.2145896
2: -4.0563812, -3.5707695, -4.0563812, -3.5707695, -0.2288604, 0.2288604
3: -12.0196257, -11.4445486, -12.0196257, -11.4445486, -0.2750299, 0.2750301
4: -2.2219746, -1.7096759, -2.2219746, -1.7096759, -0.2323757, 0.2323757
5: -9.8138056, -9.2945709, -9.8138056, -9.2945709, -0.1485732, 0.1485732
6: -6.6133256, -5.9296675, -6.6133256, -5.9296675, -0.3522463, 0.3522463
7: -3.1167550, -2.6428435, -3.1167550, -2.6428435, -0.2625113, 0.2625113
8: -2.9814658, -2.5360699, -2.9814658, -2.5360699, -0.2275560, 0.2275561
9: -12.8839951, -12.3250294, -12.8839951, -12.3250294, -0.3172929, 0.3172929

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 51

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 51

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 54.92 + 175.47 = 230.40 seconds
