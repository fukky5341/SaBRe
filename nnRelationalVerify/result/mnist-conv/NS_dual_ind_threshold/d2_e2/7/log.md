## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.259609539


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5734241, 0.5734241)
1: (-4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5444160, 0.5444160)
2: (-5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4155605, 0.4155604)
3: (-10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4519312, 0.4519312)
4: (4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5382333, 0.5382330)
5: (-7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3759611, 0.3759613)
6: (-3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5690289, 0.5690289)
7: (-6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6293364, 0.6293364)
8: (-3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4772696, 0.4772696)
9: (-6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5227687, 0.5227687)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.42 + 33.71 = 57.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.2676387, upper bound: 0.2676391

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2676213, upper bound: 0.2665052
time: 3.68 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2676386, upper bound: 0.2676390
time: 3.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.69 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.69
Output dim: 4, lower bound: -0.2676213, upper bound: 0.2665052
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.69
Output dim: 4, lower bound: -0.2676386, upper bound: 0.2676390

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -7.1057281, -6.2920442, -7.1057281, -6.2920437, -0.5734220, 0.5734119
1: -4.9336247, -4.0876627, -4.9336262, -4.0876632, -0.5444155, 0.5444164
2: -5.6065383, -4.8446727, -5.6065388, -4.8446698, -0.4155605, 0.4155604
3: -10.7152195, -9.9662924, -10.7152185, -9.9662876, -0.4519305, 0.4519308
4: 4.3862052, 5.0951376, 4.3862052, 5.0951319, -0.5382323, 0.5382385
5: -7.9719658, -7.2899961, -7.9719667, -7.2899981, -0.3759617, 0.3759611
6: -3.2702127, -2.2950273, -3.2702138, -2.2950287, -0.5690284, 0.5690286
7: -6.3882127, -5.3162727, -6.3882108, -5.3162708, -0.6293359, 0.6293337
8: -3.3521700, -2.5929770, -3.3521700, -2.5929770, -0.4772713, 0.4772696
9: -6.7731881, -5.9340925, -6.7731886, -5.9340925, -0.5227678, 0.5227578

Time for backsubstitution: 20.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2665048, upper bound: 0.2665051
time: 4.31 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2665048, upper bound: 0.2665051
time: 3.80 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -7.1057281, -6.2920442, -7.1057286, -6.2920437, -0.5734239, 0.5734107
1: -4.9336257, -4.0876632, -4.9336252, -4.0876632, -0.5444171, 0.5444160
2: -5.6065378, -4.8446698, -5.6065383, -4.8446698, -0.4155605, 0.4155611
3: -10.7152195, -9.9662857, -10.7152185, -9.9662857, -0.4519312, 0.4519324
4: 4.3862057, 5.0951319, 4.3862057, 5.0951324, -0.5382419, 0.5382333
5: -7.9719663, -7.2899961, -7.9719667, -7.2899966, -0.3759611, 0.3759623
6: -3.2702122, -2.2950277, -3.2702131, -2.2950277, -0.5690279, 0.5690284
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6293364, 0.6293354
8: -3.3521690, -2.5929761, -3.3521690, -2.5929775, -0.4772704, 0.4772716
9: -6.7731891, -5.9340935, -6.7731895, -5.9340916, -0.5227687, 0.5227562

Time for backsubstitution: 20.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2665048, upper bound: 0.2676213
time: 3.83 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2665048, upper bound: 0.2676213
time: 4.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.78 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.78
Output dim: 4, lower bound: -0.2665048, upper bound: 0.2665051
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.78
Output dim: 4, lower bound: -0.2665048, upper bound: 0.2665051
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.78
Output dim: 4, lower bound: -0.2665048, upper bound: 0.2676213
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.78
Output dim: 4, lower bound: -0.2665048, upper bound: 0.2676213

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -7.1057281, -6.2920442, -7.1057281, -6.2920442, -0.5734115, 0.5734115
1: -4.9336247, -4.0876627, -4.9336247, -4.0876627, -0.5444162, 0.5444162
2: -5.6065383, -4.8446727, -5.6065383, -4.8446727, -0.4155602, 0.4155602
3: -10.7152195, -9.9662924, -10.7152195, -9.9662924, -0.4519308, 0.4519308
4: 4.3862052, 5.0951376, 4.3862052, 5.0951376, -0.5382383, 0.5382385
5: -7.9719658, -7.2899961, -7.9719658, -7.2899961, -0.3759618, 0.3759618
6: -3.2702127, -2.2950273, -3.2702127, -2.2950273, -0.5690281, 0.5690284
7: -6.3882127, -5.3162727, -6.3882127, -5.3162727, -0.6293335, 0.6293333
8: -3.3521700, -2.5929770, -3.3521700, -2.5929770, -0.4772713, 0.4772711
9: -6.7731881, -5.9340925, -6.7731881, -5.9340925, -0.5227578, 0.5227578

Time for backsubstitution: 20.90 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1489
type: A, layer: 3, pos: 1166
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: A, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: A, layer: 3, pos: 1788

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 2371

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2601217, upper bound: 0.2612729
time: 3.93 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2616180, upper bound: 0.2613394
time: 3.76 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -7.1057281, -6.2920442, -7.1057281, -6.2920442, -0.5734234, 0.5734117
1: -4.9336247, -4.0876627, -4.9336257, -4.0876632, -0.5444157, 0.5444164
2: -5.6065383, -4.8446727, -5.6065378, -4.8446698, -0.4155605, 0.4155602
3: -10.7152195, -9.9662924, -10.7152195, -9.9662857, -0.4519303, 0.4519310
4: 4.3862052, 5.0951376, 4.3862057, 5.0951319, -0.5382333, 0.5382383
5: -7.9719658, -7.2899961, -7.9719663, -7.2899961, -0.3759618, 0.3759613
6: -3.2702127, -2.2950273, -3.2702122, -2.2950277, -0.5690284, 0.5690284
7: -6.3882127, -5.3162727, -6.3882132, -5.3162689, -0.6293366, 0.6293335
8: -3.3521700, -2.5929770, -3.3521690, -2.5929761, -0.4772711, 0.4772704
9: -6.7731881, -5.9340925, -6.7731891, -5.9340935, -0.5227687, 0.5227578

Time for backsubstitution: 21.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1489
type: A, layer: 3, pos: 1166
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: A, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: A, layer: 3, pos: 1788

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 2371

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2601217, upper bound: 0.2612729
time: 4.08 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2616180, upper bound: 0.2613394
time: 4.06 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -7.1057281, -6.2920442, -7.1057281, -6.2920442, -0.5734117, 0.5734234
1: -4.9336257, -4.0876632, -4.9336247, -4.0876627, -0.5444164, 0.5444157
2: -5.6065378, -4.8446698, -5.6065383, -4.8446727, -0.4155602, 0.4155604
3: -10.7152195, -9.9662857, -10.7152195, -9.9662924, -0.4519310, 0.4519303
4: 4.3862057, 5.0951319, 4.3862052, 5.0951376, -0.5382383, 0.5382330
5: -7.9719663, -7.2899961, -7.9719658, -7.2899961, -0.3759613, 0.3759618
6: -3.2702122, -2.2950277, -3.2702127, -2.2950273, -0.5690281, 0.5690281
7: -6.3882132, -5.3162689, -6.3882127, -5.3162727, -0.6293335, 0.6293364
8: -3.3521690, -2.5929761, -3.3521700, -2.5929770, -0.4772704, 0.4772713
9: -6.7731891, -5.9340935, -6.7731881, -5.9340925, -0.5227578, 0.5227687

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1489
type: A, layer: 3, pos: 1166
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: A, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: A, layer: 3, pos: 1788

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 2371

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2601996, upper bound: 0.2631923
time: 4.14 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2613383, upper bound: 0.2632393
time: 3.93 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -7.1057281, -6.2920442, -7.1057281, -6.2920442, -0.5734103, 0.5734103
1: -4.9336257, -4.0876632, -4.9336257, -4.0876632, -0.5444169, 0.5444171
2: -5.6065378, -4.8446698, -5.6065378, -4.8446698, -0.4155610, 0.4155608
3: -10.7152195, -9.9662857, -10.7152195, -9.9662857, -0.4519324, 0.4519324
4: 4.3862057, 5.0951319, 4.3862057, 5.0951319, -0.5382414, 0.5382414
5: -7.9719663, -7.2899961, -7.9719663, -7.2899961, -0.3759623, 0.3759623
6: -3.2702122, -2.2950277, -3.2702122, -2.2950277, -0.5690281, 0.5690279
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6293352, 0.6293352
8: -3.3521690, -2.5929761, -3.3521690, -2.5929761, -0.4772718, 0.4772718
9: -6.7731891, -5.9340935, -6.7731891, -5.9340935, -0.5227561, 0.5227560

Time for backsubstitution: 21.80 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1489
type: A, layer: 3, pos: 1166
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: A, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: A, layer: 3, pos: 1788

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 2371

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2601994, upper bound: 0.2632297
time: 3.83 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2613383, upper bound: 0.2632507
time: 3.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.59 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 4, lower bound: -0.2601217, upper bound: 0.2612729
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 4, lower bound: -0.2616180, upper bound: 0.2613394
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 4, lower bound: -0.2601217, upper bound: 0.2612729
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 4, lower bound: -0.2616180, upper bound: 0.2613394
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 4, lower bound: -0.2601996, upper bound: 0.2631923
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 4, lower bound: -0.2613383, upper bound: 0.2632393
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 4, lower bound: -0.2601994, upper bound: 0.2632297
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 4, lower bound: -0.2613383, upper bound: 0.2632507

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.0952840, -6.2921286, -7.1024680, -6.2920675, -0.5565140, 0.5685921
1: -4.9294353, -4.0876627, -4.9324484, -4.0876627, -0.5384624, 0.5426695
2: -5.6023097, -4.8446760, -5.6053486, -4.8446727, -0.4114838, 0.4144026
3: -10.7106829, -9.9676371, -10.7135811, -9.9666672, -0.4474497, 0.4475436
4: 4.4004917, 5.0951371, 4.3902144, 5.0951376, -0.5200288, 0.5329049
5: -7.9719162, -7.2963433, -7.9719510, -7.2917795, -0.3735310, 0.3674245
6: -3.2700737, -2.3039837, -3.2701735, -2.2975399, -0.5652308, 0.5566134
7: -6.3826804, -5.3162880, -6.3866138, -5.3162770, -0.6207137, 0.6268857
8: -3.3515239, -2.6022530, -3.3519907, -2.5956717, -0.4740086, 0.4673378
9: -6.7608652, -5.9344530, -6.7694325, -5.9341927, -0.5101876, 0.5190543

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: B, layer: 3, pos: 1158
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1489
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: B, layer: 3, pos: 1788

Time for candidate selection: 0.45 seconds

### Candidate
type: B, layer: 3, pos: 1747

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2572225, upper bound: 0.2508461
time: 3.97 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2572225, upper bound: 0.2590929
time: 4.07 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.1020536, -6.2817202, -7.1038232, -6.2920499, -0.5620928, 0.5902064
1: -4.9324393, -4.0870819, -4.9329829, -4.0876627, -0.5423934, 0.5453746
2: -5.6060863, -4.8418450, -5.6062407, -4.8446722, -0.4143391, 0.4183699
3: -10.7124004, -9.9597578, -10.7143927, -9.9664097, -0.4532430, 0.4470649
4: 4.3933449, 5.1024938, 4.3885183, 5.0951376, -0.5234821, 0.5526142
5: -7.9735408, -7.2944927, -7.9719639, -7.2914095, -0.3815438, 0.3691459
6: -3.2744153, -2.3000028, -3.2702060, -2.2967732, -0.5794826, 0.5575242
7: -6.3842173, -5.3156815, -6.3868222, -5.3162727, -0.6233768, 0.6342797
8: -3.3613744, -2.5932956, -3.3521123, -2.5932274, -0.4869609, 0.4721613
9: -6.7715645, -5.9220724, -6.7725592, -5.9341221, -0.5174303, 0.5315549

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: B, layer: 3, pos: 1158
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1489
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: B, layer: 3, pos: 1788

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 1747

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2592008, upper bound: 0.2509287
time: 3.82 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2592008, upper bound: 0.2592015
time: 4.33 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.0952840, -6.2921286, -7.1024709, -6.2920694, -0.5565264, 0.5685933
1: -4.9294353, -4.0876627, -4.9324484, -4.0876632, -0.5384614, 0.5426695
2: -5.6023097, -4.8446760, -5.6053486, -4.8446703, -0.4114833, 0.4144001
3: -10.7106829, -9.9676371, -10.7135792, -9.9666595, -0.4474497, 0.4475381
4: 4.4004917, 5.0951371, 4.3902130, 5.0951319, -0.5200231, 0.5329053
5: -7.9719162, -7.2963433, -7.9719501, -7.2917795, -0.3735311, 0.3674245
6: -3.2700737, -2.3039837, -3.2701743, -2.2975421, -0.5652308, 0.5566139
7: -6.3826804, -5.3162880, -6.3866148, -5.3162727, -0.6207175, 0.6268864
8: -3.3515239, -2.6022530, -3.3519902, -2.5956726, -0.4740090, 0.4673369
9: -6.7608652, -5.9344530, -6.7694225, -5.9341927, -0.5101986, 0.5190551

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: B, layer: 3, pos: 1158
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1489
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: B, layer: 3, pos: 1788

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 1747

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2571969, upper bound: 0.2505975
time: 3.92 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2571969, upper bound: 0.2588541
time: 4.06 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.1020536, -6.2817202, -7.1038218, -6.2920504, -0.5621047, 0.5902064
1: -4.9324393, -4.0870819, -4.9329829, -4.0876632, -0.5423927, 0.5453746
2: -5.6060863, -4.8418450, -5.6062393, -4.8446689, -0.4143391, 0.4183698
3: -10.7124004, -9.9597578, -10.7143927, -9.9664040, -0.4532428, 0.4470651
4: 4.3933449, 5.1024938, 4.3885174, 5.0951328, -0.5234752, 0.5526142
5: -7.9735408, -7.2944927, -7.9719629, -7.2914095, -0.3815439, 0.3691452
6: -3.2744153, -2.3000028, -3.2702062, -2.2967734, -0.5794826, 0.5575242
7: -6.3842173, -5.3156815, -6.3868217, -5.3162699, -0.6233797, 0.6342797
8: -3.3613744, -2.5932956, -3.3521132, -2.5932288, -0.4869609, 0.4721603
9: -6.7715645, -5.9220724, -6.7725611, -5.9341230, -0.5174410, 0.5315552

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: B, layer: 3, pos: 1158
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1489
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: B, layer: 3, pos: 1788

Time for candidate selection: 0.35 seconds

### Candidate
type: B, layer: 3, pos: 1747

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2608215, upper bound: 0.2506539
time: 4.06 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2608214, upper bound: 0.2589222
time: 3.83 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.0952702, -6.2921305, -7.1024685, -6.2920680, -0.5565145, 0.5686045
1: -4.9294357, -4.0876632, -4.9324484, -4.0876627, -0.5384624, 0.5426683
2: -5.6023088, -4.8446727, -5.6053491, -4.8446727, -0.4114809, 0.4144026
3: -10.7106819, -9.9676313, -10.7135811, -9.9666672, -0.4474440, 0.4475429
4: 4.4004927, 5.0951324, 4.3902144, 5.0951376, -0.5200288, 0.5328999
5: -7.9719162, -7.2963438, -7.9719510, -7.2917814, -0.3735301, 0.3674247
6: -3.2700734, -2.3039863, -3.2701735, -2.2975397, -0.5652304, 0.5566120
7: -6.3826804, -5.3162847, -6.3866138, -5.3162770, -0.6207142, 0.6268890
8: -3.3515215, -2.6022520, -3.3519897, -2.5956736, -0.4740081, 0.4673383
9: -6.7608542, -5.9344535, -6.7694330, -5.9341927, -0.5101879, 0.5190661

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: B, layer: 3, pos: 1158
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1489
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: B, layer: 3, pos: 1788

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 1747

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2573876, upper bound: 0.2525164
time: 4.02 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2573876, upper bound: 0.2607736
time: 4.04 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.1020522, -6.2817087, -7.1038218, -6.2920499, -0.5620925, 0.5902171
1: -4.9324398, -4.0870829, -4.9329824, -4.0876627, -0.5423939, 0.5453734
2: -5.6060858, -4.8418427, -5.6062422, -4.8446722, -0.4143364, 0.4183724
3: -10.7124014, -9.9597530, -10.7143927, -9.9664087, -0.4532380, 0.4470706
4: 4.3933449, 5.1024880, 4.3885193, 5.0951376, -0.5234821, 0.5526071
5: -7.9735398, -7.2944913, -7.9719639, -7.2914090, -0.3815432, 0.3691459
6: -3.2744164, -2.3000040, -3.2702065, -2.2967720, -0.5794835, 0.5575225
7: -6.3842173, -5.3156767, -6.3868237, -5.3162727, -0.6233768, 0.6342821
8: -3.3613739, -2.5932961, -3.3521128, -2.5932274, -0.4869595, 0.4721615
9: -6.7715664, -5.9220610, -6.7725601, -5.9341230, -0.5174303, 0.5315654

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: B, layer: 3, pos: 1158
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1489
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: B, layer: 3, pos: 1788

Time for candidate selection: 0.45 seconds

### Candidate
type: B, layer: 3, pos: 1747

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2589213, upper bound: 0.2525577
time: 4.09 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2589213, upper bound: 0.2608224
time: 3.86 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.0952702, -6.2921305, -7.1024704, -6.2920694, -0.5565131, 0.5685925
1: -4.9294357, -4.0876632, -4.9324484, -4.0876632, -0.5384629, 0.5426702
2: -5.6023088, -4.8446727, -5.6053491, -4.8446703, -0.4114840, 0.4144027
3: -10.7106819, -9.9676313, -10.7135801, -9.9666605, -0.4474523, 0.4475455
4: 4.4004927, 5.0951324, 4.3902140, 5.0951319, -0.5200319, 0.5329080
5: -7.9719162, -7.2963438, -7.9719505, -7.2917795, -0.3735312, 0.3674252
6: -3.2700734, -2.3039863, -3.2701743, -2.2975407, -0.5652299, 0.5566130
7: -6.3826804, -5.3162847, -6.3866143, -5.3162727, -0.6207161, 0.6268876
8: -3.3515215, -2.6022520, -3.3519888, -2.5956717, -0.4740098, 0.4673381
9: -6.7608542, -5.9344535, -6.7694225, -5.9341922, -0.5101864, 0.5190531

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: B, layer: 3, pos: 1158
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1489
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: B, layer: 3, pos: 1788

Time for candidate selection: 0.35 seconds

### Candidate
type: B, layer: 3, pos: 1747

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2574422, upper bound: 0.2525531
time: 5.94 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2574422, upper bound: 0.2608112
time: 5.48 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.1020522, -6.2817087, -7.1038237, -6.2920504, -0.5620966, 0.5902047
1: -4.9324398, -4.0870829, -4.9329824, -4.0876632, -0.5423942, 0.5453751
2: -5.6060858, -4.8418427, -5.6062403, -4.8446689, -0.4143398, 0.4183712
3: -10.7124014, -9.9597530, -10.7143936, -9.9664030, -0.4532452, 0.4470658
4: 4.3933449, 5.1024880, 4.3885174, 5.0951328, -0.5234818, 0.5526161
5: -7.9735398, -7.2944913, -7.9719629, -7.2914095, -0.3815438, 0.3691461
6: -3.2744164, -2.3000040, -3.2702062, -2.2967730, -0.5794814, 0.5575235
7: -6.3842173, -5.3156767, -6.3868227, -5.3162699, -0.6233776, 0.6342812
8: -3.3613739, -2.5932961, -3.3521137, -2.5932274, -0.4869609, 0.4721622
9: -6.7715664, -5.9220610, -6.7725625, -5.9341221, -0.5174344, 0.5315527

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2371
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 1998
type: B, layer: 3, pos: 1158
type: B, layer: 3, pos: 2118
type: B, layer: 3, pos: 2334
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1489
type: B, layer: 3, pos: 1166
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 2397
type: B, layer: 3, pos: 3109
type: B, layer: 3, pos: 1237
type: B, layer: 3, pos: 1788

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 1747

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2589338, upper bound: 0.2525731
time: 4.80 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2589338, upper bound: 0.2608325
time: 5.36 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 32.41 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2572225, upper bound: 0.2508461
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2572225, upper bound: 0.2590929
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2592008, upper bound: 0.2509287
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2592008, upper bound: 0.2592015
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2571969, upper bound: 0.2505975
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2571969, upper bound: 0.2588541
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2608215, upper bound: 0.2506539
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2608214, upper bound: 0.2589222
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2573876, upper bound: 0.2525164
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2573876, upper bound: 0.2607736
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2589213, upper bound: 0.2525577
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2589213, upper bound: 0.2608224
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2574422, upper bound: 0.2525531
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2574422, upper bound: 0.2608112
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2589338, upper bound: 0.2525731
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.41
Output dim: 4, lower bound: -0.2589338, upper bound: 0.2608325

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.1008196, -6.2820292, -7.1153684, -6.2931314, -0.5534678, 0.5861988
1: -4.9308400, -4.0870819, -4.9279838, -4.0818539, -0.5323033, 0.5330362
2: -5.6058073, -4.8461471, -5.6105571, -4.8599396, -0.3988082, 0.4156711
3: -10.7123690, -9.9670811, -10.7168560, -9.9918251, -0.4184330, 0.4308996
4: 4.4072528, 5.1024704, 4.4287081, 5.0904846, -0.5010026, 0.5006390
5: -7.9734941, -7.2964621, -7.9744492, -7.2967114, -0.3733109, 0.3657842
6: -3.2744155, -2.3136106, -3.2759385, -2.3375645, -0.5243759, 0.5322821
7: -6.3630075, -5.3156805, -6.3164244, -5.3206830, -0.5833149, 0.5514350
8: -3.3608856, -2.5960917, -3.3519888, -2.6033139, -0.4736407, 0.4664371
9: -6.7699242, -5.9230881, -6.7720666, -5.9372334, -0.5060987, 0.5237706

Time for backsubstitution: 21.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1489
type: A, layer: 3, pos: 1166
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: A, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: A, layer: 3, pos: 1788

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 2371

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2571969, upper bound: 0.2505976
time: 4.02 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2608215, upper bound: 0.2506539
time: 4.15 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.1020536, -6.2817202, -7.1056938, -6.2940855, -0.5658789, 0.5891156
1: -4.9324393, -4.0870819, -4.9321947, -4.0876632, -0.5549910, 0.5454915
2: -5.6060863, -4.8418450, -5.6065240, -4.8469467, -0.4057709, 0.4180446
3: -10.7124004, -9.9597578, -10.7152071, -9.9680853, -0.4230379, 0.4488447
4: 4.3933449, 5.1024938, 4.3876162, 5.0951209, -0.5234628, 0.5161319
5: -7.9735408, -7.2944927, -7.9719448, -7.2906418, -0.3795414, 0.3691235
6: -3.2744153, -2.3000028, -3.2702124, -2.2974260, -0.5339842, 0.5575304
7: -6.3842173, -5.3156815, -6.3849821, -5.3162689, -0.6233799, 0.5704294
8: -3.3613744, -2.5932956, -3.3519530, -2.5936546, -0.4834366, 0.4719663
9: -6.7715645, -5.9220724, -6.7731423, -5.9353852, -0.5238161, 0.5312954

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1489
type: A, layer: 3, pos: 1166
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: A, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: A, layer: 3, pos: 1788

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 3, pos: 2371

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2571970, upper bound: 0.2588542
time: 4.51 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2608214, upper bound: 0.2589222
time: 4.03 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.0952702, -6.2921305, -7.1056933, -6.2940836, -0.5603302, 0.5718608
1: -4.9294357, -4.0876632, -4.9321938, -4.0876627, -0.5510356, 0.5438206
2: -5.6023088, -4.8446727, -5.6065226, -4.8469505, -0.4029384, 0.4150515
3: -10.7106819, -9.9676313, -10.7152061, -9.9680910, -0.4179037, 0.4487267
4: 4.4004927, 5.0951324, 4.3876152, 5.0951266, -0.5200150, 0.5007505
5: -7.9719162, -7.2963438, -7.9719467, -7.2906437, -0.3731627, 0.3674116
6: -3.2700734, -2.3039863, -3.2702110, -2.2974238, -0.5227263, 0.5566597
7: -6.3826804, -5.3162847, -6.3849802, -5.3162737, -0.6207192, 0.5646005
8: -3.3515215, -2.6022520, -3.3519535, -2.5936561, -0.4730012, 0.4672401
9: -6.7608542, -5.9344535, -6.7731400, -5.9353848, -0.5166237, 0.5210855

Time for backsubstitution: 21.90 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2371
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 1158
type: A, layer: 3, pos: 2118
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 2334
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1489
type: A, layer: 3, pos: 1166
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 652
type: A, layer: 3, pos: 2397
type: A, layer: 3, pos: 3109
type: A, layer: 3, pos: 1237
type: A, layer: 3, pos: 1788

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 2371

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2573876, upper bound: 0.2607736
time: 3.97 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2573876, upper bound: 0.2607736
time: 4.27 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.1020522, -6.2817087, -7.1056933, -6.2940836, -0.5658681, 0.5891266
1: -4.9324398, -4.0870829, -4.9321938, -4.0876627, -0.5549917, 0.5454903
2: -5.6060858, -4.8418427, -5.6065226, -4.8469505, -0.4057679, 0.4180472
3: -10.7124014, -9.9597530, -10.7152061, -9.9680910, -0.4230328, 0.4488499
4: 4.3933449, 5.1024880, 4.3876152, 5.0951266, -0.5234687, 0.5161259
5: -7.9735398, -7.2944913, -7.9719467, -7.2906437, -0.3795402, 0.3691239
6: -3.2744164, -2.3000040, -3.2702110, -2.2974238, -0.5339854, 0.5575290
7: -6.3842173, -5.3156767, -6.3849802, -5.3162737, -0.6233771, 0.5704322
8: -3.3613739, -2.5932961, -3.3519535, -2.5936561, -0.4834347, 0.4719672
9: -6.7715664, -5.9220610, -6.7731400, -5.9353848, -0.5238054, 0.5313056

Time for backsubstitution: 22.77 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.13 + 546.61 = 603.74 seconds
