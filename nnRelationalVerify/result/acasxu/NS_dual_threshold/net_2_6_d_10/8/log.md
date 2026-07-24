## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 42.289434762380004


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160)
1: (-17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844)
2: (-13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863)
3: (-14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786)
4: (-11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.37 + 1.71 = 4.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -42.3021254, upper bound: 42.3021254

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2929218, upper bound: 42.2912901
time: 0.55 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.3007659, upper bound: 42.3007659
time: 0.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.33 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 3, lower bound: -42.2929218, upper bound: 42.2912901
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 3, lower bound: -42.3007659, upper bound: 42.3007659

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -64.5731506, 107.2421494, -97.5574417, 165.7690125, -230.3421478, 204.7995911
1: -11.5029459, 15.4048920, -17.7561874, 23.8874989, -35.3904457, 33.1610794
2: -8.6509132, 14.4953432, -13.4649935, 21.8683014, -30.5192146, 27.9603329
3: -9.1930447, 24.5248833, -14.3390293, 36.8165550, -46.0095978, 38.8639107
4: -7.2651896, 18.4225006, -11.3048334, 27.4657993, -34.7309837, 29.7273331

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2834459, upper bound: 42.2834459
time: 0.57 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2834459, upper bound: 42.2834459
time: 0.60 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -96.6716614, 164.1812592, -97.5574417, 165.7690125, -262.4406738, 261.7386780
1: -17.5651093, 23.6462498, -17.7561874, 23.8874989, -41.4526062, 41.4024353
2: -13.3320065, 21.6681614, -13.4649935, 21.8683014, -35.2002983, 35.1331558
3: -14.2025223, 36.4909859, -14.3390293, 36.8165550, -51.0190735, 50.8300133
4: -11.1946545, 27.2369251, -11.3048334, 27.4657993, -38.6604538, 38.5417595

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.3001212, upper bound: 42.2964495
time: 0.61 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.3002313, upper bound: 42.3002312
time: 0.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.55 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.55
Output dim: 3, lower bound: -42.2834459, upper bound: 42.2834459
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 3.55
Output dim: 3, lower bound: -42.2834459, upper bound: 42.2834459
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -42.3001212, upper bound: 42.2964495
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -42.3002313, upper bound: 42.3002312

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -87.5046997, 147.5444489, -97.5574417, 165.7690125, -253.2737122, 245.1018982
1: -15.4879665, 21.2585869, -17.7561874, 23.8874989, -39.3754616, 39.0147743
2: -11.9386606, 19.6704426, -13.4649935, 21.8683014, -33.8069611, 33.1354370
3: -12.6444101, 32.9829483, -14.3390293, 36.8165550, -49.4609604, 47.3219757
4: -9.9854441, 24.9064960, -11.3048334, 27.4657993, -37.4512444, 36.2113304

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2924891, upper bound: 42.2951158
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2986354, upper bound: 42.2951299
time: 0.59 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -96.4044495, 163.6560364, -97.5574417, 165.7690125, -262.1734009, 261.2134399
1: -17.5036869, 23.5671024, -17.7561874, 23.8874989, -41.3911858, 41.3232880
2: -13.2873392, 21.6035042, -13.4649935, 21.8683014, -35.1556282, 35.0684967
3: -14.1545620, 36.3858452, -14.3390293, 36.8165550, -50.9711113, 50.7248726
4: -11.1575108, 27.1595058, -11.3048334, 27.4657993, -38.6233101, 38.4643402

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912245, upper bound: 42.2924526
time: 0.68 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912245, upper bound: 42.2994340
time: 0.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.63 seconds
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 3, lower bound: -42.2924891, upper bound: 42.2951158
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 3, lower bound: -42.2986354, upper bound: 42.2951299
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 3, lower bound: -42.2912245, upper bound: 42.2924526
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 3, lower bound: -42.2912245, upper bound: 42.2994340

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -87.5046997, 147.5444489, -88.5253601, 147.9494629, -235.4541626, 236.0698090
1: -15.4879665, 21.2585869, -15.6403179, 21.3898697, -36.8778305, 36.8988991
2: -11.9386606, 19.6704426, -12.0042181, 19.7152786, -31.6539383, 31.6746597
3: -12.6444101, 32.9829483, -12.7411547, 33.1989632, -45.8433647, 45.7241020
4: -9.9854441, 24.9064960, -10.0738916, 24.9517231, -34.9371681, 34.9803886

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2923244, upper bound: 42.2887280
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2918428, upper bound: 42.2939090
time: 0.61 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -87.5046997, 147.5444489, -94.7869415, 160.5241547, -248.0288544, 242.3313751
1: -15.4879665, 21.2585869, -17.1680145, 23.1477604, -38.6357269, 38.4266014
2: -11.9386606, 19.6704426, -13.0390100, 21.2198639, -33.1585236, 32.7094536
3: -12.6444101, 32.9829483, -13.8729992, 35.7217789, -48.3661842, 46.8559494
4: -9.9854441, 24.9064960, -10.9444828, 26.7077332, -36.6931763, 35.8509789

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2984050, upper bound: 42.2891068
time: 0.53 seconds

## Relational analysis of NS_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2979234, upper bound: 42.2942878
time: 0.54 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -96.4044495, 163.6560364, -64.5731506, 107.2421494, -203.6465607, 228.2291870
1: -17.5036869, 23.5671024, -11.5029459, 15.4048920, -32.9085770, 35.0700493
2: -13.2873392, 21.6035042, -8.6509132, 14.4953432, -27.7826805, 30.2544174
3: -14.1545620, 36.3858452, -9.1930447, 24.5248833, -38.6794434, 45.5788879
4: -11.1575108, 27.1595058, -7.2651896, 18.4225006, -29.5800095, 34.4246941

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2728783, upper bound: 42.2715639
time: 0.71 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2901670, upper bound: 42.2915763
time: 0.58 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -96.4044495, 163.6560364, -96.6716614, 164.1812592, -260.5856628, 260.3276672
1: -17.5036869, 23.5671024, -17.5651093, 23.6462498, -41.1499290, 41.1322098
2: -13.2873392, 21.6035042, -13.3320065, 21.6681614, -34.9554939, 34.9355049
3: -14.1545620, 36.3858452, -14.2025223, 36.4909859, -50.6455460, 50.5883675
4: -11.1575108, 27.1595058, -11.1946545, 27.2369251, -38.3944359, 38.3541603

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2836046, upper bound: 42.2941455
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898092, upper bound: 42.2975328
time: 0.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.54 seconds
NS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.54
Output dim: 3, lower bound: -42.2923244, upper bound: 42.2887280
NS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.54
Output dim: 3, lower bound: -42.2918428, upper bound: 42.2939090
NS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.54
Output dim: 3, lower bound: -42.2984050, upper bound: 42.2891068
NS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.54
Output dim: 3, lower bound: -42.2979234, upper bound: 42.2942878
NS_A2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 3.54
Output dim: 3, lower bound: -42.2728783, upper bound: 42.2715639
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.54
Output dim: 3, lower bound: -42.2901670, upper bound: 42.2915763
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.54
Output dim: 3, lower bound: -42.2836046, upper bound: 42.2941455
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.54
Output dim: 3, lower bound: -42.2898092, upper bound: 42.2975328

## BFS NS instance: NS_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -86.2547913, 145.2348175, -88.5253601, 147.9494629, -234.2042236, 233.7601776
1: -15.2426262, 20.9395504, -15.6403179, 21.3898697, -36.6324883, 36.5798645
2: -11.7561359, 19.3797626, -12.0042181, 19.7152786, -31.4714146, 31.3839779
3: -12.4324656, 32.4979095, -12.7411547, 33.1989632, -45.6314278, 45.2390633
4: -9.8297701, 24.5323067, -10.0738916, 24.9517231, -34.7814941, 34.6061974

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2780628, upper bound: 42.2829776
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B1_A1_B2

### Relational analysis result of NS_A2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2884444, upper bound: 42.2875666
time: 0.63 seconds

## BFS NS instance: NS_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -86.9849243, 146.5279236, -88.5253601, 147.9494629, -234.9343719, 235.0532837
1: -15.3744869, 21.1130142, -15.6403179, 21.3898697, -36.7643547, 36.7533340
2: -11.8542423, 19.5521812, -12.0042181, 19.7152786, -31.5695210, 31.5563984
3: -12.5514717, 32.7862892, -12.7411547, 33.1989632, -45.7504349, 45.5274429
4: -9.9141016, 24.7585850, -10.0738916, 24.9517231, -34.8658257, 34.8324776

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2776379, upper bound: 42.2879435
time: 0.58 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2880195, upper bound: 42.2925324
time: 0.59 seconds

## BFS NS instance: NS_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -86.2547913, 145.2348175, -94.7869415, 160.5241547, -246.7789154, 240.0217133
1: -15.2426262, 20.9395504, -17.1680145, 23.1477604, -38.3903847, 38.1075668
2: -11.7561359, 19.3797626, -13.0390100, 21.2198639, -32.9759979, 32.4187737
3: -12.4324656, 32.4979095, -13.8729992, 35.7217789, -48.1542435, 46.3709030
4: -9.8297701, 24.5323067, -10.9444828, 26.7077332, -36.5375023, 35.4767914

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_B2_A1_A1

### Relational analysis result of NS_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2923220, upper bound: 42.2859233
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B2_A1_A2

### Relational analysis result of NS_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2923220, upper bound: 42.2891068
time: 0.58 seconds

## BFS NS instance: NS_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -86.9849243, 146.5279236, -94.7869415, 160.5241547, -247.5090637, 241.3148346
1: -15.3744869, 21.1130142, -17.1680145, 23.1477604, -38.5222473, 38.2810287
2: -11.8542423, 19.5521812, -13.0390100, 21.2198639, -33.0741005, 32.5911903
3: -12.5514717, 32.7862892, -13.8729992, 35.7217789, -48.2732506, 46.6592865
4: -9.9141016, 24.7585850, -10.9444828, 26.7077332, -36.6218338, 35.7030678

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_B2_A2_A1

### Relational analysis result of NS_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2914770, upper bound: 42.2892757
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2

### Relational analysis result of NS_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2914770, upper bound: 42.2942878
time: 0.63 seconds

## BFS NS instance: NS_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -96.2049332, 163.2863159, -64.5731506, 107.2421494, -203.4470825, 227.8594666
1: -17.4613762, 23.5143299, -11.5029459, 15.4048920, -32.8662682, 35.0172768
2: -13.2565041, 21.5573807, -8.6509132, 14.4953432, -27.7518463, 30.2082901
3: -14.1237001, 36.3091583, -9.1930447, 24.5248833, -38.6485786, 45.5022011
4: -11.1322145, 27.1045074, -7.2651896, 18.4225006, -29.5547142, 34.3696899

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2873057, upper bound: 42.2914377
time: 0.57 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2891655, upper bound: 42.2910290
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -96.4044495, 163.6560364, -83.0989075, 138.8080750, -235.2124786, 246.7549438
1: -17.5036869, 23.5671024, -14.6601601, 20.0233669, -37.5270462, 38.2272568
2: -13.2873392, 21.6035042, -11.2216139, 18.5158386, -31.8031769, 32.8251190
3: -14.1545620, 36.3858452, -11.9348688, 31.2319984, -45.3865547, 48.3207092
4: -11.1575108, 27.1595058, -9.4175386, 23.4295139, -34.5870247, 36.5770454

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2835804, upper bound: 42.2746763
time: 0.57 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2890735, upper bound: 42.2930956
time: 0.57 seconds

## BFS NS instance: NS_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -96.4044495, 163.6560364, -94.3425827, 159.7554626, -256.1598816, 257.9986267
1: -17.5036869, 23.5671024, -17.0360737, 23.0040302, -40.5077095, 40.6031761
2: -13.2873392, 21.6035042, -12.9604301, 21.1229248, -34.4102554, 34.5639343
3: -14.1545620, 36.3858452, -13.8272543, 35.6083870, -49.7629471, 50.2131004
4: -11.1575108, 27.1595058, -10.8889589, 26.5734615, -37.7309723, 38.0484619

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2968530, upper bound: 42.2914272
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2968228, upper bound: 42.2959075
time: 0.56 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.59 seconds
NS_A2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 3.59
Output dim: 3, lower bound: -42.2780628, upper bound: 42.2829776
NS_A2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 3.59
Output dim: 3, lower bound: -42.2884444, upper bound: 42.2875666
NS_A2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 3.59
Output dim: 3, lower bound: -42.2776379, upper bound: 42.2879435
NS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 3, lower bound: -42.2880195, upper bound: 42.2925324
NS_A2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 3, lower bound: -42.2923220, upper bound: 42.2859233
NS_A2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 3, lower bound: -42.2923220, upper bound: 42.2891068
NS_A2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 3, lower bound: -42.2914770, upper bound: 42.2892757
NS_A2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 3, lower bound: -42.2914770, upper bound: 42.2942878
NS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 3, lower bound: -42.2873057, upper bound: 42.2914377
NS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 3, lower bound: -42.2891655, upper bound: 42.2910290
NS_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.59
Output dim: 3, lower bound: -42.2835804, upper bound: 42.2746763
NS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 3, lower bound: -42.2890735, upper bound: 42.2930956
NS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 3, lower bound: -42.2968530, upper bound: 42.2914272
NS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 3, lower bound: -42.2968228, upper bound: 42.2959075

## BFS NS instance: NS_A2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -86.9849243, 146.5279236, -88.3262711, 147.5847321, -234.5696564, 234.8541718
1: -15.3744869, 21.1130142, -15.5988865, 21.3364964, -36.7109833, 36.7118988
2: -11.8542423, 19.5521812, -11.9733171, 19.6706505, -31.5248928, 31.5254955
3: -12.5514717, 32.7862892, -12.7100859, 33.1258659, -45.6773376, 45.4963722
4: -9.9141016, 24.7585850, -10.0485058, 24.8975468, -34.8116493, 34.8070908

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2875007, upper bound: 42.2842732
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2875007, upper bound: 42.2925324
time: 0.58 seconds

## BFS NS instance: NS_A2_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -79.7379532, 132.8270569, -94.7869415, 160.5241547, -240.2621002, 227.6139679
1: -13.8163853, 19.1164112, -17.1680145, 23.1477604, -36.9641457, 36.2844238
2: -10.7023077, 17.8857441, -13.0390100, 21.2198639, -31.9221649, 30.9247551
3: -11.2860775, 30.0438538, -13.8729992, 35.7217789, -47.0078506, 43.9168549
4: -8.9463549, 22.6970158, -10.9444828, 26.7077332, -35.6540871, 33.6414986

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B2_A1_A1_B1

### Relational analysis result of NS_A2_A1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2810635, upper bound: 42.2779990
time: 0.63 seconds

## Relational analysis of NS_A2_A1_B2_A1_A1_B2

### Relational analysis result of NS_A2_A1_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2810635, upper bound: 42.2859233
time: 0.55 seconds

## BFS NS instance: NS_A2_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -83.7960739, 140.6881104, -94.7869415, 160.5241547, -244.3202209, 235.4750366
1: -14.7467222, 20.2764530, -17.1680145, 23.1477604, -37.8944817, 37.4444656
2: -11.3771286, 18.8300724, -13.0390100, 21.2198639, -32.5969925, 31.8690834
3: -12.0087166, 31.5858917, -13.8729992, 35.7217789, -47.7304878, 45.4588890
4: -9.5078917, 23.8509693, -10.9444828, 26.7077332, -36.2156258, 34.7954521

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B2_A1_A2_B1

### Relational analysis result of NS_A2_A1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2810635, upper bound: 42.2811825
time: 0.65 seconds

## Relational analysis of NS_A2_A1_B2_A1_A2_B2

### Relational analysis result of NS_A2_A1_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2810635, upper bound: 42.2811825
time: 0.56 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -80.4889297, 134.2168732, -94.7869415, 160.5241547, -241.0130615, 229.0037689
1: -13.9753628, 19.3147945, -17.1680145, 23.1477604, -37.1231232, 36.4828110
2: -10.8173466, 18.0702553, -13.0390100, 21.2198639, -32.0372086, 31.1092644
3: -11.4179010, 30.3441277, -13.8729992, 35.7217789, -47.1396790, 44.2171249
4: -9.0446625, 22.9294643, -10.9444828, 26.7077332, -35.7523956, 33.8739471

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B2_A2_A1_A1

### Relational analysis result of NS_A2_A1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2879118, upper bound: 42.2892757
time: 0.55 seconds

## Relational analysis of NS_A2_A1_B2_A2_A1_A2

### Relational analysis result of NS_A2_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2914770, upper bound: 42.2891795
time: 0.68 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -84.6446075, 142.2179108, -94.7869415, 160.5241547, -245.1687622, 237.0048218
1: -14.9038200, 20.4824257, -17.1680145, 23.1477604, -38.0515823, 37.6504402
2: -11.4947090, 19.0322971, -13.0390100, 21.2198639, -32.7145653, 32.0713081
3: -12.1447201, 31.9245377, -13.8729992, 35.7217789, -47.8664970, 45.7975388
4: -9.6080790, 24.1117687, -10.9444828, 26.7077332, -36.3158112, 35.0562515

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B2_A2_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2802185, upper bound: 42.2863635
time: 0.61 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2802185, upper bound: 42.2863635
time: 0.55 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -96.2049332, 163.2863159, -63.4158859, 105.0801849, -201.2851257, 226.7022095
1: -17.4613762, 23.5143299, -11.2715168, 15.1195984, -32.5809746, 34.7858467
2: -13.2565041, 21.5573807, -8.4790878, 14.2271996, -27.4837036, 30.0364666
3: -14.1237001, 36.3091583, -9.0006895, 24.0783310, -38.2020264, 45.3098412
4: -11.1322145, 27.1045074, -7.1188016, 18.0738640, -29.2060776, 34.2233047

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2830799, upper bound: 42.2804996
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2836076, upper bound: 42.2892454
time: 0.56 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -96.2049332, 163.2863159, -64.0688477, 106.2767181, -202.4816437, 227.3551636
1: -17.4613762, 23.5143299, -11.3962364, 15.2665358, -32.7279129, 34.9105644
2: -13.2565041, 21.5573807, -8.5716400, 14.3827152, -27.6392193, 30.1290169
3: -14.1237001, 36.3091583, -9.1080761, 24.3377857, -38.4614754, 45.4172287
4: -11.1322145, 27.1045074, -7.1979055, 18.2798271, -29.4120407, 34.3024063

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2862393, upper bound: 42.2803362
time: 0.58 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867670, upper bound: 42.2890820
time: 0.63 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -96.2049332, 163.2863159, -83.0989075, 138.8080750, -235.0130005, 246.3852234
1: -17.4613762, 23.5143299, -14.6601601, 20.0233669, -37.4847412, 38.1744843
2: -13.2565041, 21.5573807, -11.2216139, 18.5158386, -31.7723427, 32.7789879
3: -14.1237001, 36.3091583, -11.9348688, 31.2319984, -45.3556938, 48.2440186
4: -11.1322145, 27.1045074, -9.4175386, 23.4295139, -34.5617294, 36.5220413

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2883467, upper bound: 42.2839848
time: 0.63 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2883294, upper bound: 42.2916039
time: 0.57 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -87.4028244, 145.8477936, -94.3425827, 159.7554626, -247.1582947, 240.1903687
1: -15.3881998, 21.0674973, -17.0360737, 23.0040302, -38.3922310, 38.1035690
2: -11.8258476, 19.4588623, -12.9604301, 21.1229248, -32.9487648, 32.4192924
3: -12.5528059, 32.7854881, -13.8272543, 35.6083870, -48.1611862, 46.6127434
4: -9.9264688, 24.6466999, -10.8889589, 26.5734615, -36.4999313, 35.5356560

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2905417, upper bound: 42.2913495
time: 0.54 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2957102, upper bound: 42.2907116
time: 0.58 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -93.6359329, 158.3874817, -94.3425827, 159.7554626, -253.3913879, 252.7300720
1: -16.9122810, 22.8227062, -17.0360737, 23.0040302, -39.9163132, 39.8587761
2: -12.8587856, 20.9513645, -12.9604301, 21.1229248, -33.9817085, 33.9117928
3: -13.6852341, 35.2850227, -13.8272543, 35.6083870, -49.2936172, 49.1122780
4: -10.7949686, 26.3976364, -10.8889589, 26.5734615, -37.3684311, 37.2865906

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2907386, upper bound: 42.2957122
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2959071, upper bound: 42.2951011
time: 0.54 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.60 seconds
NS_A2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2875007, upper bound: 42.2842732
NS_A2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2875007, upper bound: 42.2925324
NS_A2_A1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2810635, upper bound: 42.2779990
NS_A2_A1_B2_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2810635, upper bound: 42.2859233
NS_A2_A1_B2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2810635, upper bound: 42.2811825
NS_A2_A1_B2_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2810635, upper bound: 42.2811825
NS_A2_A1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2879118, upper bound: 42.2892757
NS_A2_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2914770, upper bound: 42.2891795
NS_A2_A1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2802185, upper bound: 42.2863635
NS_A2_A1_B2_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2802185, upper bound: 42.2863635
NS_A2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2830799, upper bound: 42.2804996
NS_A2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2836076, upper bound: 42.2892454
NS_A2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2862393, upper bound: 42.2803362
NS_A2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2867670, upper bound: 42.2890820
NS_A2_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2883467, upper bound: 42.2839848
NS_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2883294, upper bound: 42.2916039
NS_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2905417, upper bound: 42.2913495
NS_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2957102, upper bound: 42.2907116
NS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2907386, upper bound: 42.2957122
NS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 3, lower bound: -42.2959071, upper bound: 42.2951011

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -84.6446075, 142.2179108, -88.3262711, 147.5847321, -232.2293396, 230.5441895
1: -14.9038200, 20.4824257, -15.5988865, 21.3364964, -36.2403183, 36.0813103
2: -11.4947090, 19.0322971, -11.9733171, 19.6706505, -31.1653538, 31.0056152
3: -12.1447201, 31.9245377, -12.7100859, 33.1258659, -45.2705841, 44.6346245
4: -9.6080790, 24.1117687, -10.0485058, 24.8975468, -34.5056229, 34.1602669

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2776807, upper bound: 42.2836079
time: 0.58 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2776807, upper bound: 42.2836079
time: 0.58 seconds

## BFS NS instance: NS_A2_A1_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -80.0819550, 133.5184326, -94.7869415, 160.5241547, -240.6061096, 228.3053436
1: -13.9032040, 19.2114315, -17.1680145, 23.1477604, -37.0509644, 36.3794479
2: -10.7579861, 17.9829807, -13.0390100, 21.2198639, -31.9778404, 31.0219917
3: -11.3570356, 30.1943436, -13.8729992, 35.7217789, -47.0788155, 44.0673447
4: -8.9949493, 22.8206730, -10.9444828, 26.7077332, -35.7026825, 33.7651558

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B2_A2_A1_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2873686, upper bound: 42.2812553
time: 0.58 seconds

## Relational analysis of NS_A2_A1_B2_A2_A1_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2873686, upper bound: 42.2891795
time: 0.59 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -93.4259338, 158.0005493, -83.0989075, 138.8080750, -232.2340088, 241.0994415
1: -16.8677216, 22.7672234, -14.6601601, 20.0233669, -36.8910866, 37.4273796
2: -12.8262348, 20.9029083, -11.2216139, 18.5158386, -31.3420734, 32.1245232
3: -13.6521807, 35.2046432, -11.9348688, 31.2319984, -44.8841705, 47.1395073
4: -10.7682199, 26.3402443, -9.4175386, 23.4295139, -34.1977348, 35.7577820

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2849108, upper bound: 42.2912927
time: 0.64 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2873766, upper bound: 42.2910269
time: 0.59 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -87.4028244, 145.8477936, -92.6378250, 156.5611725, -243.9639893, 238.4856110
1: -15.3881998, 21.0674973, -16.6796017, 22.5772514, -37.9654503, 37.7471008
2: -11.8258476, 19.4588623, -12.7144938, 20.7148361, -32.5406799, 32.1733551
3: -12.5528059, 32.7854881, -13.5544662, 34.9128113, -47.4656105, 46.3399544
4: -9.9264688, 24.6466999, -10.6761265, 26.0916519, -36.0181198, 35.3228264

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2884198, upper bound: 42.2895961
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2888296, upper bound: 42.2905385
time: 0.59 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -87.4028244, 145.8477936, -93.6982574, 158.5012512, -245.9040833, 239.5460358
1: -15.3881998, 21.0674973, -16.8906403, 22.8317165, -38.2199173, 37.9581337
2: -11.8258476, 19.4588623, -12.8594971, 20.9729233, -32.7987709, 32.3183594
3: -12.5528059, 32.7854881, -13.7191982, 35.3537598, -47.9065628, 46.5046806
4: -9.9264688, 24.6466999, -10.8018866, 26.3992195, -36.3256874, 35.4485855

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895193, upper bound: 42.2888129
time: 0.57 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2944311, upper bound: 42.2899401
time: 0.60 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -93.6359329, 158.3874817, -92.6378250, 156.5611725, -250.1971130, 251.0252991
1: -16.9122810, 22.8227062, -16.6796017, 22.5772514, -39.4895325, 39.5023041
2: -12.8587856, 20.9513645, -12.7144938, 20.7148361, -33.5736198, 33.6658554
3: -13.6852341, 35.2850227, -13.5544662, 34.9128113, -48.5980415, 48.8394852
4: -10.7949686, 26.3976364, -10.6761265, 26.0916519, -36.8866196, 37.0737610

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2885125, upper bound: 42.2934218
time: 0.60 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2889222, upper bound: 42.2941859
time: 0.62 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -93.6359329, 158.3874817, -93.6982574, 158.5012512, -252.1371765, 252.0857391
1: -16.9122810, 22.8227062, -16.8906403, 22.8317165, -39.7439957, 39.7133369
2: -12.8587856, 20.9513645, -12.8594971, 20.9729233, -33.8317108, 33.8108597
3: -13.6852341, 35.2850227, -13.7191982, 35.3537598, -49.0389938, 49.0042114
4: -10.7949686, 26.3976364, -10.8018866, 26.3992195, -37.1941872, 37.1995239

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2896119, upper bound: 42.2926446
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2945237, upper bound: 42.2936230
time: 0.57 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.62 seconds
NS_A2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -42.2776807, upper bound: 42.2836079
NS_A2_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -42.2776807, upper bound: 42.2836079
NS_A2_A1_B2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -42.2873686, upper bound: 42.2812553
NS_A2_A1_B2_A2_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -42.2873686, upper bound: 42.2891795
NS_A2_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -42.2849108, upper bound: 42.2912927
NS_A2_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -42.2873766, upper bound: 42.2910269
NS_A2_A2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -42.2884198, upper bound: 42.2895961
NS_A2_A2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -42.2888296, upper bound: 42.2905385
NS_A2_A2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -42.2895193, upper bound: 42.2888129
NS_A2_A2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -42.2944311, upper bound: 42.2899401
NS_A2_A2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -42.2885125, upper bound: 42.2934218
NS_A2_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -42.2889222, upper bound: 42.2941859
NS_A2_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -42.2896119, upper bound: 42.2926446
NS_A2_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -42.2945237, upper bound: 42.2936230

## BFS NS instance: NS_A2_A2_B2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -93.4259338, 158.0005493, -81.9699326, 136.6876678, -230.1136017, 239.9704437
1: -16.8677216, 22.7672234, -14.4385004, 19.7311974, -36.5989189, 37.2057228
2: -12.8262348, 20.9029083, -11.0543537, 18.2350540, -31.0612869, 31.9572620
3: -13.6521807, 35.2046432, -11.7414398, 30.7629166, -44.4150963, 46.9460793
4: -10.7682199, 26.3402443, -9.2744932, 23.0855579, -33.8537788, 35.6147385

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2830658, upper bound: 42.2910705
time: 0.63 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2827672, upper bound: 42.2896576
time: 0.67 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2819183, upper bound: 42.2841053
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2819183, upper bound: 42.2912927
time: 0.59 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -93.4259338, 158.0005493, -82.5924072, 137.8040466, -231.2299805, 240.5929565
1: -16.8677216, 22.7672234, -14.5478477, 19.8819771, -36.7496948, 37.3150711
2: -12.8262348, 20.9029083, -11.1384420, 18.3935986, -31.2198334, 32.0413513
3: -13.6521807, 35.2046432, -11.8439617, 31.0268993, -44.6790771, 47.0486031
4: -10.7682199, 26.3402443, -9.3472223, 23.2826786, -34.0508995, 35.6874657

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2853240, upper bound: 42.2908071
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2856242, upper bound: 42.2879715
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -87.4028244, 145.8477936, -88.6641006, 149.6990662, -237.1018829, 234.5118713
1: -15.3881998, 21.0674973, -15.9057827, 21.5329742, -36.9211731, 36.9732819
2: -11.8258476, 19.4588623, -12.1232758, 19.8422451, -31.6680927, 31.5821381
3: -12.5528059, 32.7854881, -12.9621143, 33.4779701, -46.0307693, 45.7476044
4: -9.9264688, 24.6466999, -10.1904202, 24.9979305, -34.9244003, 34.8371201

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2864464, upper bound: 42.2893536
time: 0.64 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2863712, upper bound: 42.2895961
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2884198, upper bound: 42.2895532
time: 0.63 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -87.4028244, 145.8477936, -92.1425400, 155.6616058, -243.0644226, 237.9903259
1: -15.3881998, 21.0674973, -16.5772934, 22.4400234, -37.8282242, 37.6447906
2: -11.8258476, 19.4588623, -12.6382818, 20.6000175, -32.4258614, 32.0971451
3: -12.5528059, 32.7854881, -13.4760761, 34.7245331, -47.2773323, 46.2615662
4: -9.9264688, 24.6466999, -10.6126490, 25.9519749, -35.8784447, 35.2593498

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2863781, upper bound: 42.2905381
time: 0.65 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2863781, upper bound: 42.2905385
time: 0.58 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -87.4028244, 145.8477936, -89.5531693, 151.3232117, -238.7260437, 235.4009094
1: -15.3881998, 21.0674973, -16.0815525, 21.7491570, -37.1373558, 37.1490479
2: -11.8258476, 19.4588623, -12.2451448, 20.0624218, -31.8882637, 31.7040043
3: -12.5528059, 32.7854881, -13.1025219, 33.8527489, -46.4055519, 45.8880081
4: -9.9264688, 24.6466999, -10.2978811, 25.2627754, -35.1892433, 34.9445801

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2876468, upper bound: 42.2885587
time: 0.55 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2876064, upper bound: 42.2857432
time: 0.61 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -87.4028244, 145.8477936, -93.2011032, 157.6007385, -245.0035706, 239.0488892
1: -15.3881998, 21.0674973, -16.7877560, 22.6946125, -38.0828133, 37.8552551
2: -11.8258476, 19.4588623, -12.7831774, 20.8579979, -32.6838379, 32.2420387
3: -12.5528059, 32.7854881, -13.6406221, 35.1651154, -47.7179184, 46.4261093
4: -9.9264688, 24.6466999, -10.7381010, 26.2595024, -36.1859665, 35.3847961

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2923825, upper bound: 42.2899401
time: 0.60 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2944311, upper bound: 42.2898821
time: 0.61 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -93.6359329, 158.3874817, -88.6641006, 149.6990662, -243.3349915, 247.0515747
1: -16.9122810, 22.8227062, -15.9057827, 21.5329742, -38.4452553, 38.7284889
2: -12.8587856, 20.9513645, -12.1232758, 19.8422451, -32.7010307, 33.0746384
3: -13.6852341, 35.2850227, -12.9621143, 33.4779701, -47.1632004, 48.2471390
4: -10.7949686, 26.3976364, -10.1904202, 24.9979305, -35.7929001, 36.5880585

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2866909, upper bound: 42.2931621
time: 0.61 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2871891, upper bound: 42.2888676
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2773286, upper bound: 42.2850789
time: 0.61 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2773286, upper bound: 42.2934218
time: 0.58 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -93.6359329, 158.3874817, -92.1425400, 155.6616058, -249.2975464, 250.5300293
1: -16.9122810, 22.8227062, -16.5772934, 22.4400234, -39.3523026, 39.4000015
2: -12.8587856, 20.9513645, -12.6382818, 20.6000175, -33.4588013, 33.5896454
3: -13.6852341, 35.2850227, -13.4760761, 34.7245331, -48.4097633, 48.7610970
4: -10.7949686, 26.3976364, -10.6126490, 25.9519749, -36.7469406, 37.0102844

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2864707, upper bound: 42.2941859
time: 0.58 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2864707, upper bound: 42.2941859
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -93.6359329, 158.3874817, -89.5531693, 151.3232117, -244.9591370, 247.9406433
1: -16.9122810, 22.8227062, -16.0815525, 21.7491570, -38.6614380, 38.9042587
2: -12.8587856, 20.9513645, -12.2451448, 20.0624218, -32.9212074, 33.1965103
3: -13.6852341, 35.2850227, -13.1025219, 33.8527489, -47.5379791, 48.3875427
4: -10.7949686, 26.3976364, -10.2978811, 25.2627754, -36.0577431, 36.6955185

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2878913, upper bound: 42.2923849
time: 0.66 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2878509, upper bound: 42.2890393
time: 0.62 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -93.6359329, 158.3874817, -93.2011032, 157.6007385, -251.2366638, 251.5885925
1: -16.9122810, 22.8227062, -16.7877560, 22.6946125, -39.6068916, 39.6104584
2: -12.8587856, 20.9513645, -12.7831774, 20.8579979, -33.7167778, 33.7345428
3: -13.6852341, 35.2850227, -13.6406221, 35.1651154, -48.8503494, 48.9256439
4: -10.7949686, 26.3976364, -10.7381010, 26.2595024, -37.0544662, 37.1357346

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2910666, upper bound: 42.2909526
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2945237, upper bound: 42.2936230
time: 0.64 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 4.21 seconds
NS_A2_A2_B2_B1_A2_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2819183, upper bound: 42.2841053
NS_A2_A2_B2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2819183, upper bound: 42.2912927
NS_A2_A2_B2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2853240, upper bound: 42.2908071
NS_A2_A2_B2_B1_A2_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2856242, upper bound: 42.2879715
NS_A2_A2_B2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2863712, upper bound: 42.2895961
NS_A2_A2_B2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2884198, upper bound: 42.2895532
NS_A2_A2_B2_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2863781, upper bound: 42.2905381
NS_A2_A2_B2_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2863781, upper bound: 42.2905385
NS_A2_A2_B2_B2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2876468, upper bound: 42.2885587
NS_A2_A2_B2_B2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2876064, upper bound: 42.2857432
NS_A2_A2_B2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2923825, upper bound: 42.2899401
NS_A2_A2_B2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2944311, upper bound: 42.2898821
NS_A2_A2_B2_B2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2773286, upper bound: 42.2850789
NS_A2_A2_B2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2773286, upper bound: 42.2934218
NS_A2_A2_B2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2864707, upper bound: 42.2941859
NS_A2_A2_B2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2864707, upper bound: 42.2941859
NS_A2_A2_B2_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2878913, upper bound: 42.2923849
NS_A2_A2_B2_B2_A2_B2_B1_B2, status: Status.VERIFIED, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2878509, upper bound: 42.2890393
NS_A2_A2_B2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2910666, upper bound: 42.2909526
NS_A2_A2_B2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -42.2945237, upper bound: 42.2936230

## BFS NS instance: NS_A2_A2_B2_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -91.0604095, 153.5643921, -81.9699326, 136.6876678, -227.7480621, 235.5343170
1: -16.3386250, 22.1269188, -14.4385004, 19.7311974, -36.0698242, 36.5654182
2: -12.4554577, 20.3593998, -11.0543537, 18.2350540, -30.6905117, 31.4137535
3: -13.2774229, 34.3214493, -11.7414398, 30.7629166, -44.0403404, 46.0628815
4: -10.4602585, 25.6781368, -9.2744932, 23.0855579, -33.5458145, 34.9526291

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2819183, upper bound: 42.2816887
time: 0.56 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2819183, upper bound: 42.2912927
time: 0.58 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -93.4259338, 158.0005493, -80.0433121, 133.5206604, -226.9465790, 238.0438080
1: -16.8677216, 22.7672234, -14.0841780, 19.2168427, -36.0845642, 36.8514023
2: -12.8262348, 20.9029083, -10.7669411, 17.8435993, -30.6698341, 31.6698494
3: -13.6521807, 35.2046432, -11.4657202, 30.1192169, -43.7713890, 46.6703568
4: -10.7682199, 26.3402443, -9.0393419, 22.5770435, -33.3452644, 35.3795853

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2830811, upper bound: 42.2857730
time: 0.60 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2853240, upper bound: 42.2908071
time: 0.60 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -84.8850632, 142.8439484, -88.6641006, 149.6990662, -234.5841064, 231.5080566
1: -15.0730915, 20.6303272, -15.9057827, 21.5329742, -36.6060638, 36.5361099
2: -11.5441523, 19.0132561, -12.1232758, 19.8422451, -31.3863983, 31.1365280
3: -12.2610378, 32.0082817, -12.9621143, 33.4779701, -45.7390060, 44.9703979
4: -9.6835575, 24.1226692, -10.1904202, 24.9979305, -34.6814880, 34.3130875

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2825883, upper bound: 42.2868314
time: 0.66 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2825883, upper bound: 42.2868313
time: 0.60 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -86.8999100, 144.9782104, -88.6641006, 149.6990662, -236.5989685, 233.6423035
1: -15.2939510, 20.9383564, -15.9057827, 21.5329742, -36.8269196, 36.8441391
2: -11.7524242, 19.3501606, -12.1232758, 19.8422451, -31.5946693, 31.4734325
3: -12.4774256, 32.5995598, -12.9621143, 33.4779701, -45.9553947, 45.5616760
4: -9.8652716, 24.5129929, -10.1904202, 24.9979305, -34.8632011, 34.7034149

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2846119, upper bound: 42.2866780
time: 0.63 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2846119, upper bound: 42.2895532
time: 0.59 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -87.4028244, 145.8477936, -83.4950333, 138.5245209, -225.9273376, 229.3427887
1: -15.3881998, 21.0674973, -14.5451698, 20.0094662, -35.3976631, 35.6126671
2: -11.8258476, 19.4588623, -11.2166576, 18.5587902, -30.3846359, 30.6755199
3: -12.5528059, 32.7854881, -11.9125910, 31.3172188, -43.8700218, 44.6980782
4: -9.9264688, 24.6466999, -9.4196997, 23.5287876, -33.4552574, 34.0663986

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2834677, upper bound: 42.2902988
time: 0.56 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2863781, upper bound: 42.2856380
time: 0.60 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2863781, upper bound: 42.2905381
time: 0.63 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -87.4028244, 145.8477936, -89.3648911, 150.3539581, -237.7567749, 235.2126312
1: -15.3881998, 21.0674973, -15.9811735, 21.6939602, -37.0821571, 37.0486717
2: -11.8258476, 19.4588623, -12.2069817, 19.9434853, -31.7693310, 31.6658401
3: -12.5528059, 32.7854881, -13.0029697, 33.6144409, -46.1672440, 45.7884598
4: -9.9264688, 24.6466999, -10.2478046, 25.1873417, -35.1138115, 34.8945045

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2843295, upper bound: 42.2905385
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2863781, upper bound: 42.2904714
time: 0.61 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -84.8850632, 142.8439484, -93.2011032, 157.6007385, -242.4858093, 236.0450439
1: -15.0730915, 20.6303272, -16.7877560, 22.6946125, -37.7677002, 37.4180756
2: -11.5441523, 19.0132561, -12.7831774, 20.8579979, -32.4021454, 31.7964325
3: -12.2610378, 32.0082817, -13.6406221, 35.1651154, -47.4261551, 45.6489029
4: -9.6835575, 24.1226692, -10.7381010, 26.2595024, -35.9430580, 34.8607635

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2880341, upper bound: 42.2898556
time: 0.64 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2880341, upper bound: 42.2899401
time: 0.66 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -86.8999100, 144.9782104, -93.2011032, 157.6007385, -244.5006409, 238.1793213
1: -15.2939510, 20.9383564, -16.7877560, 22.6946125, -37.9885597, 37.7261047
2: -11.7524242, 19.3501606, -12.7831774, 20.8579979, -32.6104202, 32.1333313
3: -12.4774256, 32.5995598, -13.6406221, 35.1651154, -47.6425400, 46.2401810
4: -9.8652716, 24.5129929, -10.7381010, 26.2595024, -36.1247749, 35.2510872

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2900828, upper bound: 42.2898113
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2900828, upper bound: 42.2898821
time: 0.59 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -91.3446655, 154.1026306, -88.6641006, 149.6990662, -241.0437164, 242.7667236
1: -16.3998375, 22.2045193, -15.9057827, 21.5329742, -37.9328117, 38.1103020
2: -12.5002213, 20.4263268, -12.1232758, 19.8422451, -32.3424683, 32.5496025
3: -13.3225193, 34.4321175, -12.9621143, 33.4779701, -46.8004913, 47.3942337
4: -10.4969482, 25.7585354, -10.1904202, 24.9979305, -35.4948807, 35.9489555

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2730727, upper bound: 42.2901928
time: 0.58 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2730727, upper bound: 42.2934218
time: 0.61 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -93.6359329, 158.3874817, -83.4950333, 138.5245209, -232.1604614, 241.8824768
1: -16.9122810, 22.8227062, -14.5451698, 20.0094662, -36.9217453, 37.3678741
2: -12.8587856, 20.9513645, -11.2166576, 18.5587902, -31.4175758, 32.1680222
3: -13.6852341, 35.2850227, -11.9125910, 31.3172188, -45.0024490, 47.1976128
4: -10.7949686, 26.3976364, -9.4196997, 23.5287876, -34.3237572, 35.8173370

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2837121, upper bound: 42.2939236
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2864707, upper bound: 42.2882813
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2864707, upper bound: 42.2941859
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -93.6359329, 158.3874817, -89.3648911, 150.3539581, -243.9898987, 247.7523346
1: -16.9122810, 22.8227062, -15.9811735, 21.6939602, -38.6062355, 38.8038788
2: -12.8587856, 20.9513645, -12.2069817, 19.9434853, -32.8022690, 33.1583443
3: -13.6852341, 35.2850227, -13.0029697, 33.6144409, -47.2996712, 48.2879944
4: -10.7949686, 26.3976364, -10.2478046, 25.1873417, -35.9823112, 36.6454391

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2837121, upper bound: 42.2939236
time: 0.70 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2864707, upper bound: 42.2882813
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2864707, upper bound: 42.2941859
time: 0.69 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -93.6359329, 158.3874817, -87.0657578, 147.0174103, -240.6533508, 245.4532166
1: -16.9122810, 22.8227062, -15.5937700, 21.0869789, -37.9992599, 38.4164696
2: -12.8587856, 20.9513645, -11.8726358, 19.5100956, -32.3688774, 32.8240013
3: -13.6852341, 35.2850227, -12.7230291, 32.9481850, -46.6334190, 48.0080528
4: -10.7949686, 26.3976364, -9.9938087, 24.5715256, -35.3664932, 36.3914452

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2855633, upper bound: 42.2880088
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2855633, upper bound: 42.2923849
time: 0.66 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -90.9440155, 154.9081116, -93.2011032, 157.6007385, -248.5447540, 248.1092224
1: -16.5433254, 22.3302612, -16.7877560, 22.6946125, -39.2379379, 39.1180191
2: -12.5446377, 20.4681129, -12.7831774, 20.8579979, -33.4026299, 33.2512894
3: -13.3658085, 34.4641609, -13.6406221, 35.1651154, -48.5309219, 48.1047821
4: -10.5271111, 25.7950706, -10.7381010, 26.2595024, -36.7866020, 36.5331688

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2867183, upper bound: 42.2908988
time: 0.58 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2867183, upper bound: 42.2909526
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -93.1277313, 157.5046387, -93.2011032, 157.6007385, -250.7284698, 250.7057495
1: -16.8167305, 22.6931000, -16.7877560, 22.6946125, -39.5113411, 39.4808464
2: -12.7847471, 20.8404922, -12.7831774, 20.8579979, -33.6427345, 33.6236610
3: -13.6093645, 35.0940666, -13.6406221, 35.1651154, -48.7744751, 48.7346878
4: -10.7330494, 26.2626457, -10.7381010, 26.2595024, -36.9925461, 37.0007477

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2901754, upper bound: 42.2935817
time: 0.63 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2901754, upper bound: 42.2936231
time: 0.67 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 3.91 seconds
NS_A2_A2_B2_B1_A2_A2_B1_A2_A1, status: Status.VERIFIED, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2819183, upper bound: 42.2816887
NS_A2_A2_B2_B1_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2819183, upper bound: 42.2912927
NS_A2_A2_B2_B1_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2830811, upper bound: 42.2857730
NS_A2_A2_B2_B1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2853240, upper bound: 42.2908071
NS_A2_A2_B2_B2_A1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2825883, upper bound: 42.2868314
NS_A2_A2_B2_B2_A1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2825883, upper bound: 42.2868313
NS_A2_A2_B2_B2_A1_B1_B1_A2_B1, status: Status.VERIFIED, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2846119, upper bound: 42.2866780
NS_A2_A2_B2_B2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2846119, upper bound: 42.2895532
NS_A2_A2_B2_B2_A1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2863781, upper bound: 42.2856380
NS_A2_A2_B2_B2_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2863781, upper bound: 42.2905381
NS_A2_A2_B2_B2_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2843295, upper bound: 42.2905385
NS_A2_A2_B2_B2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2863781, upper bound: 42.2904714
NS_A2_A2_B2_B2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2880341, upper bound: 42.2898556
NS_A2_A2_B2_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2880341, upper bound: 42.2899401
NS_A2_A2_B2_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2900828, upper bound: 42.2898113
NS_A2_A2_B2_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2900828, upper bound: 42.2898821
NS_A2_A2_B2_B2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2730727, upper bound: 42.2901928
NS_A2_A2_B2_B2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2730727, upper bound: 42.2934218
NS_A2_A2_B2_B2_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2864707, upper bound: 42.2882813
NS_A2_A2_B2_B2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2864707, upper bound: 42.2941859
NS_A2_A2_B2_B2_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2864707, upper bound: 42.2882813
NS_A2_A2_B2_B2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2864707, upper bound: 42.2941859
NS_A2_A2_B2_B2_A2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2855633, upper bound: 42.2880088
NS_A2_A2_B2_B2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2855633, upper bound: 42.2923849
NS_A2_A2_B2_B2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2867183, upper bound: 42.2908988
NS_A2_A2_B2_B2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2867183, upper bound: 42.2909526
NS_A2_A2_B2_B2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2901754, upper bound: 42.2935817
NS_A2_A2_B2_B2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 3.91
Output dim: 3, lower bound: -42.2901754, upper bound: 42.2936231

## BFS NS instance: NS_A2_A2_B2_B1_A2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -90.4525681, 152.3227386, -81.9699326, 136.6876678, -227.1402283, 234.2926483
1: -16.1940403, 21.9570465, -14.4385004, 19.7311974, -35.9252396, 36.3955460
2: -12.3555279, 20.2106838, -11.0543537, 18.2350540, -30.5905800, 31.2650375
3: -13.1695490, 34.0686455, -11.7414398, 30.7629166, -43.9324646, 45.8100777
4: -10.3757124, 25.5065479, -9.2744932, 23.0855579, -33.4612694, 34.7810364

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2812889, upper bound: 42.2882768
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2812889, upper bound: 42.2912927
time: 0.58 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -92.9164734, 157.1148376, -80.0433121, 133.5206604, -226.4370728, 237.1580963
1: -16.7718735, 22.6372280, -14.0841780, 19.2168427, -35.9887161, 36.7214050
2: -12.7519312, 20.7917538, -10.7669411, 17.8435993, -30.5955257, 31.5586910
3: -13.5760298, 35.0131302, -11.4657202, 30.1192169, -43.6952438, 46.4788437
4: -10.7060995, 26.2049026, -9.0393419, 22.5770435, -33.2831421, 35.2442436

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2835681, upper bound: 42.2891427
time: 0.60 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2819418, upper bound: 42.2838355
time: 0.63 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2819418, upper bound: 42.2908071
time: 0.67 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -86.8999100, 144.9782104, -88.3857269, 149.1660614, -236.0659790, 233.3638916
1: -15.2939510, 20.9383564, -15.8434944, 21.4530296, -36.7469788, 36.7818375
2: -11.7524242, 19.3501606, -12.0780725, 19.7768211, -31.5292454, 31.4282284
3: -12.4774256, 32.5995598, -12.9137735, 33.3715286, -45.8489532, 45.5133324
4: -9.8652716, 24.5129929, -10.1523829, 24.9195442, -34.7848129, 34.6653748

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2846119, upper bound: 42.2849004
time: 0.63 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2846119, upper bound: 42.2895532
time: 0.60 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -86.7966232, 144.6442871, -83.4950333, 138.5245209, -225.3211365, 228.1392822
1: -15.2470188, 20.8972378, -14.5451698, 20.0094662, -35.2564774, 35.4424057
2: -11.7266321, 19.3189316, -11.2166576, 18.5587902, -30.2854233, 30.5355892
3: -12.4446182, 32.5540771, -11.9125910, 31.3172188, -43.7618370, 44.4666672
4: -9.8431940, 24.4764023, -9.4196997, 23.5287876, -33.3719826, 33.8961029

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2831478, upper bound: 42.2881205
time: 0.60 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2831478, upper bound: 42.2905668
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -84.8850632, 142.8439484, -89.3648911, 150.3539581, -235.2390137, 232.2088318
1: -15.0730915, 20.6303272, -15.9811735, 21.6939602, -36.7670517, 36.6114998
2: -11.5441523, 19.0132561, -12.2069817, 19.9434853, -31.4876366, 31.2202358
3: -12.2610378, 32.0082817, -13.0029697, 33.6144409, -45.8754768, 45.0112534
4: -9.6835575, 24.1226692, -10.2478046, 25.1873417, -34.8708992, 34.3704758

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2861686, upper bound: 42.2902991
time: 0.61 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2828160, upper bound: 42.2879233
time: 0.61 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2828160, upper bound: 42.2905385
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -86.8999100, 144.9782104, -89.3648911, 150.3539581, -237.2538757, 234.3430786
1: -15.2939510, 20.9383564, -15.9811735, 21.6939602, -36.9879074, 36.9195290
2: -11.7524242, 19.3501606, -12.2069817, 19.9434853, -31.6959095, 31.5571423
3: -12.4774256, 32.5995598, -13.0029697, 33.6144409, -46.0918655, 45.6025314
4: -9.8652716, 24.5129929, -10.2478046, 25.1873417, -35.0526123, 34.7607956

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2868000, upper bound: 42.2900929
time: 0.58 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2848386, upper bound: 42.2877847
time: 0.74 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2848386, upper bound: 42.2904714
time: 0.70 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -84.8850632, 142.8439484, -84.3954315, 140.2661591, -225.1511993, 227.2393646
1: -15.0730915, 20.6303272, -14.7315826, 20.2469959, -35.3200874, 35.3619080
2: -11.5441523, 19.0132561, -11.3513021, 18.7788887, -30.3230400, 30.3645573
3: -12.2610378, 32.0082817, -12.0709467, 31.6762772, -43.9373131, 44.0792236
4: -9.6835575, 24.1226692, -9.5331059, 23.8062115, -33.4897652, 33.6557693

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2852271, upper bound: 42.2895915
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2872432, upper bound: 42.2877597
time: 0.58 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -84.8850632, 142.8439484, -90.5276947, 152.5182190, -237.4032898, 233.3716431
1: -15.0730915, 20.6303272, -16.2183990, 21.9806728, -37.0537643, 36.8487244
2: -11.5441523, 19.0132561, -12.3713493, 20.2314262, -31.7755775, 31.3846054
3: -12.2610378, 32.0082817, -13.1873102, 34.1032562, -46.3642960, 45.1955910
4: -9.6835575, 24.1226692, -10.3883753, 25.5288086, -35.2123642, 34.5110397

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2852271, upper bound: 42.2896841
time: 0.56 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2872432, upper bound: 42.2881512
time: 0.60 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -86.8999100, 144.9782104, -84.3954315, 140.2661591, -227.1660767, 229.3736267
1: -15.2939510, 20.9383564, -14.7315826, 20.2469959, -35.5409470, 35.6699333
2: -11.7524242, 19.3501606, -11.3513021, 18.7788887, -30.5313129, 30.7014618
3: -12.4774256, 32.5995598, -12.0709467, 31.6762772, -44.1537018, 44.6705055
4: -9.8652716, 24.5129929, -9.5331059, 23.8062115, -33.6714821, 34.0460968

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2858585, upper bound: 42.2894117
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2878746, upper bound: 42.2875861
time: 0.63 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -86.8999100, 144.9782104, -90.5276947, 152.5182190, -239.4181213, 235.5059052
1: -15.2939510, 20.9383564, -16.2183990, 21.9806728, -37.2746239, 37.1567535
2: -11.7524242, 19.3501606, -12.3713493, 20.2314262, -31.9838486, 31.7215099
3: -12.4774256, 32.5995598, -13.1873102, 34.1032562, -46.5806808, 45.7868690
4: -9.8652716, 24.5129929, -10.3883753, 25.5288086, -35.3940811, 34.9013634

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2858585, upper bound: 42.2894884
time: 0.65 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2878746, upper bound: 42.2879722
time: 0.63 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -91.3446655, 154.1026306, -81.5810471, 137.1954193, -228.5400848, 235.6836395
1: -16.3998375, 22.2045193, -14.3554096, 19.6946011, -36.0944366, 36.5599289
2: -12.5002213, 20.4263268, -11.0495148, 18.3449078, -30.8451290, 31.4758396
3: -13.3225193, 34.4321175, -11.7288227, 30.7977791, -44.1202965, 46.1609421
4: -10.4969482, 25.7585354, -9.2493372, 23.2041225, -33.7010651, 35.0078735

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 18

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -91.3446655, 154.1026306, -88.3857269, 149.1660614, -240.5107269, 242.4883118
1: -16.3998375, 22.2045193, -15.8434944, 21.4530296, -37.8528671, 38.0480042
2: -12.5002213, 20.4263268, -12.0780725, 19.7768211, -32.2770386, 32.5043869
3: -13.3225193, 34.4321175, -12.9137735, 33.3715286, -46.6940460, 47.3458900
4: -10.4969482, 25.7585354, -10.1523829, 24.9195442, -35.4164886, 35.9109192

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -93.0009155, 157.1097260, -83.4950333, 138.5245209, -231.5254364, 240.6047516
1: -16.7634201, 22.6461029, -14.5451698, 20.0094662, -36.7728767, 37.1912727
2: -12.7555485, 20.7973099, -11.2166576, 18.5587902, -31.3143387, 32.0139694
3: -13.5742331, 35.0242691, -11.9125910, 31.3172188, -44.8914528, 46.9368591
4: -10.7076902, 26.2193909, -9.4196997, 23.5287876, -34.2364769, 35.6390915

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2830296, upper bound: 42.2914935
time: 0.65 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2830296, upper bound: 42.2942377
time: 0.65 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -93.0009155, 157.1097260, -89.3648911, 150.3539581, -243.3548737, 246.4745941
1: -16.7634201, 22.6461029, -15.9811735, 21.6939602, -38.4573746, 38.6272774
2: -12.7555485, 20.7973099, -12.2069817, 19.9434853, -32.6990318, 33.0042839
3: -13.5742331, 35.0242691, -13.0029697, 33.6144409, -47.1886749, 48.0272369
4: -10.7076902, 26.2193909, -10.2478046, 25.1873417, -35.8950310, 36.4671898

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2841957, upper bound: 42.2913168
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2841957, upper bound: 42.2941859
time: 0.61 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -93.0940323, 157.3884735, -87.0657578, 147.0174103, -240.1114502, 244.4542084
1: -16.7982273, 22.6704025, -15.5937700, 21.0869789, -37.8852005, 38.2641640
2: -12.7739630, 20.8244514, -11.8726358, 19.5100956, -32.2840576, 32.6970863
3: -13.5986776, 35.0776711, -12.7230291, 32.9481850, -46.5468636, 47.8006897
4: -10.7245684, 26.2429047, -9.9938087, 24.5715256, -35.2960930, 36.2367134

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2797343, upper bound: 42.2842975
time: 0.61 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2797343, upper bound: 42.2923849
time: 0.65 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -90.9440155, 154.9081116, -84.3954315, 140.2661591, -231.2101440, 239.3035126
1: -16.5433254, 22.3302612, -14.7315826, 20.2469959, -36.7903214, 37.0618439
2: -12.5446377, 20.4681129, -11.3513021, 18.7788887, -31.3235264, 31.8194160
3: -13.3658085, 34.4641609, -12.0709467, 31.6762772, -45.0420837, 46.5351028
4: -10.5271111, 25.7950706, -9.5331059, 23.8062115, -34.3333206, 35.3281784

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2842776, upper bound: 42.2906821
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2862936, upper bound: 42.2888649
time: 0.61 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -90.9440155, 154.9081116, -90.5276947, 152.5182190, -243.4622345, 245.4358063
1: -16.5433254, 22.3302612, -16.2183990, 21.9806728, -38.5239983, 38.5486603
2: -12.5446377, 20.4681129, -12.3713493, 20.2314262, -32.7760620, 32.8394623
3: -13.3658085, 34.4641609, -13.1873102, 34.1032562, -47.4690628, 47.6514702
4: -10.5271111, 25.7950706, -10.3883753, 25.5288086, -36.0559158, 36.1834412

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2842776, upper bound: 42.2907374
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2862936, upper bound: 42.2890116
time: 0.65 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -93.1277313, 157.5046387, -84.3954315, 140.2661591, -233.3938751, 241.9000549
1: -16.8167305, 22.6931000, -14.7315826, 20.2469959, -37.0637283, 37.4246750
2: -12.7847471, 20.8404922, -11.3513021, 18.7788887, -31.5636292, 32.1917953
3: -13.6093645, 35.0940666, -12.0709467, 31.6762772, -45.2856331, 47.1650124
4: -10.7330494, 26.2626457, -9.5331059, 23.8062115, -34.5392609, 35.7957535

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2861030, upper bound: 42.2933163
time: 0.63 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2881191, upper bound: 42.2915412
time: 0.67 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -93.1277313, 157.5046387, -90.5276947, 152.5182190, -245.6459503, 248.0323334
1: -16.8167305, 22.6931000, -16.2183990, 21.9806728, -38.7974014, 38.9114952
2: -12.7847471, 20.8404922, -12.3713493, 20.2314262, -33.0161667, 33.2118378
3: -13.6093645, 35.0940666, -13.1873102, 34.1032562, -47.7126198, 48.2813759
4: -10.7330494, 26.2626457, -10.3883753, 25.5288086, -36.2618561, 36.6510201

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2861030, upper bound: 42.2933510
time: 0.68 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2881191, upper bound: 42.2917414
time: 0.60 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 4.47 seconds
NS_A2_A2_B2_B1_A2_A2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2812889, upper bound: 42.2882768
NS_A2_A2_B2_B1_A2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2812889, upper bound: 42.2912927
NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2819418, upper bound: 42.2838355
NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2819418, upper bound: 42.2908071
NS_A2_A2_B2_B2_A1_B1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2846119, upper bound: 42.2849004
NS_A2_A2_B2_B2_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2846119, upper bound: 42.2895532
NS_A2_A2_B2_B2_A1_B1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2831478, upper bound: 42.2881205
NS_A2_A2_B2_B2_A1_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2831478, upper bound: 42.2905668
NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2828160, upper bound: 42.2879233
NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2828160, upper bound: 42.2905385
NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2848386, upper bound: 42.2877847
NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2848386, upper bound: 42.2904714
NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2852271, upper bound: 42.2895915
NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B2, status: Status.VERIFIED, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2872432, upper bound: 42.2877597
NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2852271, upper bound: 42.2896841
NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2872432, upper bound: 42.2881512
NS_A2_A2_B2_B2_A1_B2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2858585, upper bound: 42.2894117
NS_A2_A2_B2_B2_A1_B2_B2_A2_B1_B2, status: Status.VERIFIED, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2878746, upper bound: 42.2875861
NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2858585, upper bound: 42.2894884
NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2878746, upper bound: 42.2879722
NS_A2_A2_B2_B2_A2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2830296, upper bound: 42.2914935
NS_A2_A2_B2_B2_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2830296, upper bound: 42.2942377
NS_A2_A2_B2_B2_A2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2841957, upper bound: 42.2913168
NS_A2_A2_B2_B2_A2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2841957, upper bound: 42.2941859
NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2797343, upper bound: 42.2842975
NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2797343, upper bound: 42.2923849
NS_A2_A2_B2_B2_A2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2842776, upper bound: 42.2906821
NS_A2_A2_B2_B2_A2_B2_B2_A1_B1_B2, status: Status.VERIFIED, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2862936, upper bound: 42.2888649
NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2842776, upper bound: 42.2907374
NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2862936, upper bound: 42.2890116
NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2861030, upper bound: 42.2933163
NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2881191, upper bound: 42.2915412
NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2861030, upper bound: 42.2933510
NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 10, time: 4.47
Output dim: 3, lower bound: -42.2881191, upper bound: 42.2917414

## BFS NS instance: NS_A2_A2_B2_B1_A2_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -90.4525681, 152.3227386, -81.7026215, 136.1762390, -226.6287842, 234.0253296
1: -16.1940403, 21.9570465, -14.3791399, 19.6513538, -35.8453903, 36.3361855
2: -12.3555279, 20.2106838, -11.0096960, 18.1710701, -30.5265980, 31.2203770
3: -13.1695490, 34.0686455, -11.6930180, 30.6598282, -43.8293724, 45.7616501
4: -10.3757124, 25.5065479, -9.2374439, 23.0073299, -33.3830414, 34.7439919

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -90.5781097, 152.7267761, -80.0433121, 133.5206604, -224.0987701, 232.7700500
1: -16.2484379, 22.0034943, -14.0841780, 19.2168427, -35.4652786, 36.0876694
2: -12.3848839, 20.2541962, -10.7669411, 17.8435993, -30.2284832, 31.0211315
3: -13.2052479, 34.1399460, -11.4657202, 30.1192169, -43.3244629, 45.6056595
4: -10.4013500, 25.5499916, -9.0393419, 22.5770435, -32.9783936, 34.5893250

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2819418, upper bound: 42.2891365
time: 0.60 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2819418, upper bound: 42.2908071
time: 0.70 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -86.3044510, 143.7938538, -88.3857269, 149.1660614, -235.4705200, 232.1795349
1: -15.1550951, 20.7709827, -15.8434944, 21.4530296, -36.6081238, 36.6144714
2: -11.6547279, 19.2128544, -12.0780725, 19.7768211, -31.4315491, 31.2909260
3: -12.3708429, 32.3723221, -12.9137735, 33.3715286, -45.7423706, 45.2860947
4: -9.7832890, 24.3459301, -10.1523829, 24.9195442, -34.7028236, 34.4983139

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -86.7966232, 144.6442871, -83.2398834, 138.0283813, -224.8250122, 227.8841400
1: -15.2470188, 20.8972378, -14.4873590, 19.9329128, -35.1799278, 35.3845978
2: -11.7266321, 19.3189316, -11.1734695, 18.4990540, -30.2256851, 30.4923992
3: -12.4446182, 32.5540771, -11.8658409, 31.2217979, -43.6664162, 44.4199181
4: -9.8431940, 24.4764023, -9.3839664, 23.4540882, -33.2972794, 33.8603668

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -84.8850632, 142.8439484, -89.1015244, 149.8371277, -234.7221832, 231.9454651
1: -15.0730915, 20.6303272, -15.9201679, 21.6159534, -36.6890450, 36.5504951
2: -11.5441523, 19.0132561, -12.1627722, 19.8799591, -31.4241104, 31.1760235
3: -12.2610378, 32.0082817, -12.9553623, 33.5115700, -45.7726059, 44.9636459
4: -9.6835575, 24.1226692, -10.2110338, 25.1117325, -34.7952881, 34.3337021

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2828160, upper bound: 42.2758795
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2828160, upper bound: 42.2905385
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -86.8999100, 144.9782104, -89.1015244, 149.8371277, -236.7370300, 234.0797272
1: -15.2939510, 20.9383564, -15.9201679, 21.6159534, -36.9099045, 36.8585205
2: -11.7524242, 19.3501606, -12.1627722, 19.8799591, -31.6323833, 31.5129280
3: -12.4774256, 32.5995598, -12.9553623, 33.5115700, -45.9889946, 45.5549240
4: -9.8652716, 24.5129929, -10.2110338, 25.1117325, -34.9770012, 34.7240257

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2848386, upper bound: 42.2856380
time: 0.60 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2848386, upper bound: 42.2904714
time: 0.61 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -84.8850632, 142.8439484, -82.0962372, 136.4613342, -221.3463745, 224.9401855
1: -15.0730915, 20.6303272, -14.3257895, 19.6442547, -34.7173462, 34.9561157
2: -11.5441523, 19.0132561, -11.0176544, 18.2742100, -29.8183632, 30.0309048
3: -12.2610378, 32.0082817, -11.7343740, 30.8480568, -43.1090927, 43.7426567
4: -9.6835575, 24.1226692, -9.2570906, 23.1745682, -32.8581238, 33.3797607

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2796825, upper bound: 42.2872158
time: 0.60 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2796825, upper bound: 42.2895915
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -84.8850632, 142.8439484, -87.9216309, 148.0810547, -232.9661255, 230.7655792
1: -15.0730915, 20.6303272, -15.7168617, 21.2975502, -36.3706436, 36.3471832
2: -11.5441523, 19.0132561, -11.9880047, 19.6617718, -31.2059250, 31.0012608
3: -12.2610378, 32.0082817, -12.7977123, 33.1714859, -45.4325180, 44.8059921
4: -9.6835575, 24.1226692, -10.0706625, 24.8198299, -34.5033798, 34.1933250

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2858636, upper bound: 42.2872709
time: 0.56 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2858636, upper bound: 42.2896841
time: 0.62 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -86.8999100, 144.9782104, -87.9216309, 148.0810547, -234.9809570, 232.8998260
1: -15.2939510, 20.9383564, -15.7168617, 21.2975502, -36.5914993, 36.6552124
2: -11.7524242, 19.3501606, -11.9880047, 19.6617718, -31.4141960, 31.3381653
3: -12.4774256, 32.5995598, -12.7977123, 33.1714859, -45.6489105, 45.3972702
4: -9.8652716, 24.5129929, -10.0706625, 24.8198299, -34.6850967, 34.5836563

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903852, upper bound: 42.2853929
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903852, upper bound: 42.2879722
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -93.0009155, 157.1097260, -77.2513580, 128.4279327, -221.4288483, 234.3610687
1: -16.7634201, 22.6461029, -13.3062553, 18.4512310, -35.2146454, 35.9523582
2: -12.7555485, 20.7973099, -10.3209362, 17.3387566, -30.0943050, 31.1182442
3: -13.5742331, 35.0242691, -10.8948164, 29.1438980, -42.7181320, 45.9190865
4: -10.7076902, 26.2193909, -8.6302862, 22.0090847, -32.7167740, 34.8496742

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 18

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -93.0009155, 157.1097260, -83.2398834, 138.0283813, -231.0292969, 240.3495789
1: -16.7634201, 22.6461029, -14.4873590, 19.9329128, -36.6963272, 37.1334610
2: -12.7555485, 20.7973099, -11.1734695, 18.4990540, -31.2546024, 31.9707794
3: -13.5742331, 35.0242691, -11.8658409, 31.2217979, -44.7960320, 46.8901100
4: -10.7076902, 26.2193909, -9.3839664, 23.4540882, -34.1617775, 35.6033516

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -93.0009155, 157.1097260, -81.5826950, 136.6465759, -229.6474915, 238.6924133
1: -16.7634201, 22.6461029, -14.2859154, 19.6738014, -36.4372177, 36.9320183
2: -12.7555485, 20.7973099, -11.0320835, 18.3300400, -31.0855885, 31.8293877
3: -13.5742331, 35.0242691, -11.6521606, 30.7736416, -44.3478737, 46.6764297
4: -10.7076902, 26.2193909, -9.2217064, 23.2243195, -33.9320107, 35.4410896

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -93.0009155, 157.1097260, -89.1015244, 149.8371277, -242.8380432, 246.2112427
1: -16.7634201, 22.6461029, -15.9201679, 21.6159534, -38.3793716, 38.5662689
2: -12.7555485, 20.7973099, -12.1627722, 19.8799591, -32.6355019, 32.9600754
3: -13.5742331, 35.0242691, -12.9553623, 33.5115700, -47.0858040, 47.9796295
4: -10.7076902, 26.2193909, -10.2110338, 25.1117325, -35.8194199, 36.4304199

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -90.8773956, 153.2505493, -87.0657578, 147.0174103, -237.8947906, 240.3163147
1: -16.3028336, 22.0738659, -15.5937700, 21.0869789, -37.3898010, 37.6676331
2: -12.4277458, 20.3174801, -11.8726358, 19.5100956, -31.9378395, 32.1901169
3: -13.2480392, 34.2541199, -12.7230291, 32.9481850, -46.1962166, 46.9771500
4: -10.4366226, 25.6259880, -9.9938087, 24.5715256, -35.0081482, 35.6197929

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2768648, upper bound: 42.2893678
time: 0.69 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2768648, upper bound: 42.2822118
time: 0.66 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -90.9440155, 154.9081116, -82.0962372, 136.4613342, -227.4053040, 237.0043488
1: -16.5433254, 22.3302612, -14.3257895, 19.6442547, -36.1875801, 36.6560516
2: -12.5446377, 20.4681129, -11.0176544, 18.2742100, -30.8188457, 31.4857674
3: -13.3658085, 34.4641609, -11.7343740, 30.8480568, -44.2138672, 46.1985359
4: -10.5271111, 25.7950706, -9.2570906, 23.1745682, -33.7016754, 35.0521622

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B1_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2787329, upper bound: 42.2882094
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B1_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2787329, upper bound: 42.2906821
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -90.9440155, 154.9081116, -87.9216309, 148.0810547, -239.0250702, 242.8297272
1: -16.5433254, 22.3302612, -15.7168617, 21.2975502, -37.8408737, 38.0471230
2: -12.5446377, 20.4681129, -11.9880047, 19.6617718, -32.2064056, 32.4561157
3: -13.3658085, 34.4641609, -12.7977123, 33.1714859, -46.5372925, 47.2618713
4: -10.5271111, 25.7950706, -10.0706625, 24.8198299, -35.3469391, 35.8657341

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2819002, upper bound: 42.2890782
time: 0.63 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2819002, upper bound: 42.2907374
time: 0.60 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -93.1277313, 157.5046387, -82.0962372, 136.4613342, -229.5890503, 239.6008759
1: -16.8167305, 22.6931000, -14.3257895, 19.6442547, -36.4609833, 37.0188866
2: -12.7847471, 20.8404922, -11.0176544, 18.2742100, -31.0589523, 31.8581448
3: -13.6093645, 35.0940666, -11.7343740, 30.8480568, -44.4574127, 46.8284416
4: -10.7330494, 26.2626457, -9.2570906, 23.1745682, -33.9076157, 35.5197372

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2805584, upper bound: 42.2906979
time: 0.60 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2805584, upper bound: 42.2933153
time: 0.63 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -93.1277313, 157.5046387, -88.2308044, 147.4689331, -240.5966644, 245.7354279
1: -16.8167305, 22.6931000, -15.5448284, 21.2680531, -38.0847855, 38.2379265
2: -12.7847471, 20.8404922, -11.9595490, 19.6884747, -32.4732170, 32.8000336
3: -13.6093645, 35.0940666, -12.7068768, 33.1561699, -46.7655258, 47.8009415
4: -10.7330494, 26.2626457, -10.0352783, 24.9623375, -35.6953888, 36.2979240

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2881192, upper bound: 42.2893386
time: 0.67 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2881192, upper bound: 42.2915412
time: 0.63 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -93.1277313, 157.5046387, -87.9216309, 148.0810547, -241.2087860, 245.4262695
1: -16.8167305, 22.6931000, -15.7168617, 21.2975502, -38.1142807, 38.4099579
2: -12.7847471, 20.8404922, -11.9880047, 19.6617718, -32.4465103, 32.8284950
3: -13.6093645, 35.0940666, -12.7977123, 33.1714859, -46.7808380, 47.8917770
4: -10.7330494, 26.2626457, -10.0706625, 24.8198299, -35.5528755, 36.3333092

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2855633, upper bound: 42.2890525
time: 0.63 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2855633, upper bound: 42.2933510
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -93.1277313, 157.5046387, -94.5020218, 160.2931061, -253.4208069, 252.0066528
1: -16.8167305, 22.6931000, -17.1207695, 23.0468655, -39.8635902, 39.8138580
2: -12.7847471, 20.8404922, -13.0060663, 21.1949730, -33.9797134, 33.8465500
3: -13.6093645, 35.0940666, -13.8478804, 35.6746445, -49.2839966, 48.9419479
4: -10.7330494, 26.2626457, -10.9132090, 26.7059250, -37.4389725, 37.1758537

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2922793, upper bound: 42.2895149
time: 0.64 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2922793, upper bound: 42.2917414
time: 0.60 seconds

## Summary of splitting at layer (split count: 10)
- Time for NS candidates: 3.92 seconds
NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2819418, upper bound: 42.2891365
NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2819418, upper bound: 42.2908071
NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2828160, upper bound: 42.2758795
NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2828160, upper bound: 42.2905385
NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2848386, upper bound: 42.2856380
NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2848386, upper bound: 42.2904714
NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2796825, upper bound: 42.2872158
NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2796825, upper bound: 42.2895915
NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2858636, upper bound: 42.2872709
NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2858636, upper bound: 42.2896841
NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2903852, upper bound: 42.2853929
NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2903852, upper bound: 42.2879722
NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A2_B1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2768648, upper bound: 42.2893678
NS_A2_A2_B2_B2_A2_B2_B1_B1_A2_A2_B2, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2768648, upper bound: 42.2822118
NS_A2_A2_B2_B2_A2_B2_B2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2787329, upper bound: 42.2882094
NS_A2_A2_B2_B2_A2_B2_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2787329, upper bound: 42.2906821
NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2819002, upper bound: 42.2890782
NS_A2_A2_B2_B2_A2_B2_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2819002, upper bound: 42.2907374
NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2805584, upper bound: 42.2906979
NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2805584, upper bound: 42.2933153
NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2881192, upper bound: 42.2893386
NS_A2_A2_B2_B2_A2_B2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2881192, upper bound: 42.2915412
NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2855633, upper bound: 42.2890525
NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2855633, upper bound: 42.2933510
NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2922793, upper bound: 42.2895149
NS_A2_A2_B2_B2_A2_B2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 3, lower bound: -42.2922793, upper bound: 42.2917414

## BFS NS instance: NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -94.5771027, 160.5606689, -80.0433121, 133.5206604, -228.0977173, 240.6039429
1: -17.1568527, 23.0788498, -14.0841780, 19.2168427, -36.3736954, 37.1630287
2: -13.0247002, 21.2243557, -10.7669411, 17.8435993, -30.8682976, 31.9912910
3: -13.8702536, 35.7220802, -11.4657202, 30.1192169, -43.9894676, 47.1878014
4: -10.9304094, 26.7359867, -9.0393419, 22.5770435, -33.5074539, 35.7753296

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2824681, upper bound: 42.2878682
time: 0.61 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2824681, upper bound: 42.2908070
time: 0.66 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -84.3076172, 141.7448273, -89.1015244, 149.8371277, -234.1447449, 230.8463440
1: -14.9501991, 20.4733162, -15.9201679, 21.6159534, -36.5661507, 36.3934860
2: -11.4534969, 18.8836308, -12.1627722, 19.8799591, -31.3334503, 31.0463963
3: -12.1613541, 31.7922535, -12.9553623, 33.5115700, -45.6729126, 44.7476082
4: -9.6074724, 23.9618416, -10.2110338, 25.1117325, -34.7192039, 34.1728706

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -86.3044510, 143.7938538, -89.1015244, 149.8371277, -236.1415710, 232.8953705
1: -15.1550951, 20.7709827, -15.9201679, 21.6159534, -36.7710495, 36.6911507
2: -11.6547279, 19.2128544, -12.1627722, 19.8799591, -31.5346870, 31.3756218
3: -12.3708429, 32.3723221, -12.9553623, 33.5115700, -45.8824120, 45.3276825
4: -9.7832890, 24.3459301, -10.2110338, 25.1117325, -34.8950157, 34.5569572

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -84.8850632, 142.8439484, -81.5728073, 135.5687103, -220.4537659, 224.4167480
1: -15.0730915, 20.6303272, -14.2327576, 19.5081463, -34.5812378, 34.8630791
2: -11.5441523, 19.0132561, -10.9409389, 18.1610985, -29.7052498, 29.9541893
3: -12.2610378, 32.0082817, -11.6562309, 30.6558037, -42.9168396, 43.6645126
4: -9.6835575, 24.1226692, -9.1934338, 23.0329037, -32.7164574, 33.3161011

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2796825, upper bound: 42.2814987
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2796825, upper bound: 42.2895915
time: 0.60 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -84.8850632, 142.8439484, -87.6896973, 147.6228943, -232.5079651, 230.5336456
1: -15.0730915, 20.6303272, -15.6626148, 21.2287178, -36.3018074, 36.2929420
2: -11.5441523, 19.0132561, -11.9489822, 19.6055260, -31.1496773, 30.9622326
3: -12.2610378, 32.0082817, -12.7553072, 33.0799561, -45.3409958, 44.7635880
4: -9.6835575, 24.1226692, -10.0381088, 24.7531376, -34.4366951, 34.1607780

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2858636, upper bound: 42.2815495
time: 0.64 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2858636, upper bound: 42.2896841
time: 0.66 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -84.4340210, 140.7474365, -87.9216309, 148.0810547, -232.5150757, 228.6690674
1: -14.8187695, 20.2887669, -15.7168617, 21.2975502, -36.1163177, 36.0056305
2: -11.3886156, 18.8024158, -11.9880047, 19.6617718, -31.0503883, 30.7904205
3: -12.1103401, 31.6957226, -12.7977123, 33.1714859, -45.2818184, 44.4934273
4: -9.5642939, 23.8396435, -10.0706625, 24.8198299, -34.3841209, 33.9103012

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2864950, upper bound: 42.2845033
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2864950, upper bound: 42.2870072
time: 0.63 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_B2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -91.0545349, 152.9489136, -87.9216309, 148.0810547, -239.1355896, 240.8705444
1: -16.2224369, 22.0489273, -15.7168617, 21.2975502, -37.5199890, 37.7657890
2: -12.4127588, 20.3420067, -11.9880047, 19.6617718, -32.0745239, 32.3300095
3: -13.1609745, 34.2095032, -12.7977123, 33.1714859, -46.3324471, 47.0072136
4: -10.4118710, 25.7416153, -10.0706625, 24.8198299, -35.2316971, 35.8122787

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 30

Time for candidate selection: 0.20 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.08 + 416.11 = 420.19 seconds
